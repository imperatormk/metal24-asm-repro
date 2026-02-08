; ModuleID = 'reverse_kernel_v4'
; Use ONE threadgroup buffer with offset for A and B
source_filename = "reverse_kernel_v4"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%event_t = type opaque

; Single threadgroup buffer: [0..31] = A, [32..63] = B
@tg_buf = internal addrspace(3) global [64 x float] undef, align 4

define void @gemm_kernel(
    float addrspace(1)* noundef "air-buffer-no-alias" %A,
    float addrspace(1)* noundef "air-buffer-no-alias" %B,
    float addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %C,
    i32 noundef %tid,
    i32 noundef %simd_lane,
    i32 noundef %tg_idx
) local_unnamed_addr #0 {
entry:
  %ev = alloca %event_t addrspace(3)*, align 8
  %ev_i8 = bitcast %event_t addrspace(3)** %ev to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %ev_i8) #3

  ; Tile offset = tg_idx * 32
  %tg64 = zext i32 %tg_idx to i64
  %tile_offset = shl i64 %tg64, 5

  ; A destination = tg_buf[0]
  %A_tg = getelementptr [64 x float], [64 x float] addrspace(3)* @tg_buf, i64 0, i64 0
  %A_tg_i8 = bitcast float addrspace(3)* %A_tg to i8 addrspace(3)*
  ; A source
  %A_src = getelementptr float, float addrspace(1)* %A, i64 %tile_offset
  %A_src_i8 = bitcast float addrspace(1)* %A_src to i8 addrspace(1)*
  ; Async copy A: 32 floats to tg_buf[0..31]
  %eventA = call %event_t addrspace(3)* @air.simdgroup_async_copy_1d.p3i8.p1i8(i64 4, i64 32, i8 addrspace(3)* %A_tg_i8, i8 addrspace(1)* %A_src_i8, i64 0)
  store %event_t addrspace(3)* %eventA, %event_t addrspace(3)** %ev
  call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %ev)

  ; B destination = tg_buf[32]
  %B_tg = getelementptr [64 x float], [64 x float] addrspace(3)* @tg_buf, i64 0, i64 32
  %B_tg_i8 = bitcast float addrspace(3)* %B_tg to i8 addrspace(3)*
  ; B source
  %B_src = getelementptr float, float addrspace(1)* %B, i64 %tile_offset
  %B_src_i8 = bitcast float addrspace(1)* %B_src to i8 addrspace(1)*
  ; Async copy B: 32 floats to tg_buf[32..63]
  %eventB = call %event_t addrspace(3)* @air.simdgroup_async_copy_1d.p3i8.p1i8(i64 4, i64 32, i8 addrspace(3)* %B_tg_i8, i8 addrspace(1)* %B_src_i8, i64 0)
  store %event_t addrspace(3)* %eventB, %event_t addrspace(3)** %ev
  call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %ev)

  call void @air.wg.barrier(i32 2, i32 1)

  ; Call visible function: pass pointers to A and B portions
  call void @compute_tile.MTL_VISIBLE_FN_REF(
    float addrspace(3)* %A_tg,
    float addrspace(3)* %B_tg,
    float addrspace(1)* %C,
    i32 %tid,
    i32 %simd_lane
  ) #4

  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %ev_i8) #3
  ret void
}

declare %event_t addrspace(3)* @air.simdgroup_async_copy_1d.p3i8.p1i8(i64, i64, i8 addrspace(3)*, i8 addrspace(1)*, i64) #1
declare void @air.wait_simdgroup_events(i32, %event_t addrspace(3)**) #1
declare void @air.wg.barrier(i32, i32) #1
declare void @compute_tile.MTL_VISIBLE_FN_REF(float addrspace(3)*, float addrspace(3)*, float addrspace(1)*, i32, i32) local_unnamed_addr section "air.externally_defined"
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

attributes #0 = { convergent mustprogress nounwind willreturn "frame-pointer"="none" "min-legal-vector-width"="0" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { argmemonly mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { nounwind }
attributes #4 = { nobuiltin nounwind "no-builtins" }

!air.kernel = !{!0}
!llvm.module.flags = !{!8, !9, !10, !11, !12, !13, !14}
!air.compile_options = !{!15, !16, !17}
!air.visible_function_references = !{!18}
!llvm.ident = !{!19}
!air.version = !{!20}
!air.language_version = !{!21}
!air.source_file_name = !{!22}

!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32, i32, i32)* @gemm_kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7, !23}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"A"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"B"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"C"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!7 = !{i32 4, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_lane"}
!23 = !{i32 5, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tg_idx"}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"air.max_device_buffers", i32 31}
!10 = !{i32 7, !"air.max_constant_buffers", i32 31}
!11 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!12 = !{i32 7, !"air.max_textures", i32 128}
!13 = !{i32 7, !"air.max_read_write_textures", i32 8}
!14 = !{i32 7, !"air.max_samplers", i32 16}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"air.visible_function_reference", void (float addrspace(3)*, float addrspace(3)*, float addrspace(1)*, i32, i32)* @compute_tile.MTL_VISIBLE_FN_REF, !"compute_tile"}
!19 = !{!"Apple metal version 32023.850 (metalfe-32023.850)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"reverse_kernel_v4.ll"}
