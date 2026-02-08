; ModuleID = 'copy_2d_kernel'
; 2D async copy roundtrip test:
; 1. Device->TG: load 8x4 tile from 8x16 source (strided read)
; 2. Visible fn processes tile in threadgroup
; 3. TG->Device: write 8x4 tile to 8x16 dest (strided write)
; Tests all 4 variants of air.simdgroup_async_copy needed for flash attention.
source_filename = "copy_2d_kernel"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%event_t = type opaque

; Threadgroup buffer: 32 floats (8x4 tile)
@tg_buf = internal addrspace(3) global [32 x float] undef, align 4

define void @copy_2d_roundtrip_kernel(
    float addrspace(1)* noundef "air-buffer-no-alias" %src,
    float addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %dst,
    i32 noundef %tid,
    i32 noundef %simd_lane,
    i32 noundef %tg_idx
) local_unnamed_addr #0 {
entry:
  %ev = alloca %event_t addrspace(3)*, align 8
  %ev_i8 = bitcast %event_t addrspace(3)** %ev to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %ev_i8) #3

  %tg_ptr = getelementptr [32 x float], [32 x float] addrspace(3)* @tg_buf, i64 0, i64 0
  %tg_i8 = bitcast float addrspace(3)* %tg_ptr to i8 addrspace(3)*
  %src_i8 = bitcast float addrspace(1)* %src to i8 addrspace(1)*
  %dst_i8 = bitcast float addrspace(1)* %dst to i8 addrspace(1)*

  ; Step 1: 2D async copy DEVICE -> THREADGROUP
  ; Copy 8 rows x 4 cols from source with leading dim = 16
  ; into threadgroup with leading dim = 4 (packed)
  ; Params: sizeof(T)=4, alignof(T)=4,
  ;   dst, dst_elements_per_row=4, dst_leading_dim_scale=1, dst_tile=<4,8>,
  ;   src, src_elements_per_row=16, src_leading_dim_scale=1, src_tile=<4,8>,
  ;   offset=<0,0>, clamp=0
  %ev_load = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
    i64 4, i64 4,
    i8 addrspace(3)* %tg_i8, i64 4, i64 1, <2 x i64> <i64 4, i64 8>,
    i8 addrspace(1)* %src_i8, i64 16, i64 1, <2 x i64> <i64 4, i64 8>,
    <2 x i64> zeroinitializer, i32 0
  )
  store %event_t addrspace(3)* %ev_load, %event_t addrspace(3)** %ev
  call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %ev)
  call void @air.wg.barrier(i32 2, i32 1)

  ; Step 2: Call visible function to transform the tile in-place
  call void @process_tile.MTL_VISIBLE_FN_REF(
    float addrspace(3)* %tg_ptr,
    i32 %simd_lane
  ) #4

  call void @air.wg.barrier(i32 2, i32 1)

  ; Step 3: 2D async copy THREADGROUP -> DEVICE
  ; Write 8x4 packed tile back to 8x16 destination (strided write)
  %ev_store = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p1i8.p3i8(
    i64 4, i64 4,
    i8 addrspace(1)* %dst_i8, i64 16, i64 1, <2 x i64> <i64 4, i64 8>,
    i8 addrspace(3)* %tg_i8, i64 4, i64 1, <2 x i64> <i64 4, i64 8>,
    <2 x i64> zeroinitializer, i32 0
  )
  store %event_t addrspace(3)* %ev_store, %event_t addrspace(3)** %ev
  call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %ev)

  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %ev_i8) #3
  ret void
}

; Device->TG (read)
declare %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(i64, i64, i8 addrspace(3)*, i64, i64, <2 x i64>, i8 addrspace(1)*, i64, i64, <2 x i64>, <2 x i64>, i32) #1
; TG->Device (write)
declare %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p1i8.p3i8(i64, i64, i8 addrspace(1)*, i64, i64, <2 x i64>, i8 addrspace(3)*, i64, i64, <2 x i64>, <2 x i64>, i32) #1
declare void @air.wait_simdgroup_events(i32, %event_t addrspace(3)**) #1
declare void @air.wg.barrier(i32, i32) #1
declare void @process_tile.MTL_VISIBLE_FN_REF(float addrspace(3)*, i32) local_unnamed_addr section "air.externally_defined"
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

!0 = !{void (float addrspace(1)*, float addrspace(1)*, i32, i32, i32)* @copy_2d_roundtrip_kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!6 = !{i32 3, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_lane"}
!7 = !{i32 4, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tg_idx"}
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
!18 = !{!"air.visible_function_reference", void (float addrspace(3)*, i32)* @process_tile.MTL_VISIBLE_FN_REF, !"process_tile"}
!19 = !{!"Apple metal version 32023.850 (metalfe-32023.850)"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
!22 = !{!"copy_2d_kernel.ll"}
