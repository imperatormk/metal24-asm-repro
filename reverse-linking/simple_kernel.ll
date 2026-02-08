; ModuleID = 'reverse_kernel'
source_filename = "reverse_kernel"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%event_t = type opaque

; Threadgroup buffer for async copy
@tg_buf = internal addrspace(3) global [128 x float] undef, align 4

; The kernel entry point: contains async copy (must be in kernel body) + calls visible function
define void @reverse_kernel(float addrspace(1)* noundef "air-buffer-no-alias" %src, float addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %dst, i32 noundef %tid, i32 noundef %simd_lane) local_unnamed_addr #0 {
entry:
  %ev = alloca %event_t addrspace(3)*, align 8
  %ev_i8 = bitcast %event_t addrspace(3)** %ev to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %ev_i8) #3

  ; === ASYNC COPY: device -> threadgroup ===
  ; Compute source offset: (tid / 32) * 32
  %tid64 = zext i32 %tid to i64
  %grp = udiv i64 %tid64, 32
  %base = mul i64 %grp, 32
  %src_ptr = getelementptr float, float addrspace(1)* %src, i64 %base
  %src_i8 = bitcast float addrspace(1)* %src_ptr to i8 addrspace(1)*
  %dst_tg = bitcast [128 x float] addrspace(3)* @tg_buf to i8 addrspace(3)*

  ; Call the WORKING async copy intrinsic (directly in kernel body!)
  %event = call %event_t addrspace(3)* @air.simdgroup_async_copy_1d.p3i8.p1i8(i64 4, i64 32, i8 addrspace(3)* %dst_tg, i8 addrspace(1)* %src_i8, i64 0)
  store %event_t addrspace(3)* %event, %event_t addrspace(3)** %ev
  call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %ev)
  call void @air.wg.barrier(i32 2, i32 1)

  ; === VISIBLE FUNCTION CALL: transform the async-copied data ===
  ; Read from threadgroup
  %lane64 = zext i32 %simd_lane to i64
  %tg_elem = getelementptr [128 x float], [128 x float] addrspace(3)* @tg_buf, i64 0, i64 %lane64
  %raw_val = load float, float addrspace(3)* %tg_elem

  ; Call the JIT-compiled visible function to transform the value
  %transformed = call float @transform_value.MTL_VISIBLE_FN_REF(float %raw_val) #4

  ; Write result to device
  %dst_ptr = getelementptr float, float addrspace(1)* %dst, i64 %tid64
  store float %transformed, float addrspace(1)* %dst_ptr

  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %ev_i8) #3
  ret void
}

; Async copy intrinsics (the WORKING typed-pointer names)
declare %event_t addrspace(3)* @air.simdgroup_async_copy_1d.p3i8.p1i8(i64, i64, i8 addrspace(3)*, i8 addrspace(1)*, i64) #1
declare void @air.wait_simdgroup_events(i32, %event_t addrspace(3)**) #1
declare void @air.wg.barrier(i32, i32) #1

; Visible function reference (externally defined, linked at pipeline time)
declare float @transform_value.MTL_VISIBLE_FN_REF(float) local_unnamed_addr section "air.externally_defined"

; LLVM intrinsics
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

attributes #0 = { convergent mustprogress nounwind willreturn "frame-pointer"="none" "min-legal-vector-width"="0" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { argmemonly mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { nounwind }
attributes #4 = { nobuiltin nounwind "no-builtins" }

!air.kernel = !{!0}
!llvm.module.flags = !{!7, !8, !9, !10, !11, !12, !13}
!air.compile_options = !{!14, !15, !16}
!air.visible_function_references = !{!17}
!llvm.ident = !{!18}
!air.version = !{!19}
!air.language_version = !{!20}
!air.source_file_name = !{!21}

!0 = !{void (float addrspace(1)*, float addrspace(1)*, i32, i32)* @reverse_kernel, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!6 = !{i32 3, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_lane"}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 7, !"air.max_device_buffers", i32 31}
!9 = !{i32 7, !"air.max_constant_buffers", i32 31}
!10 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!11 = !{i32 7, !"air.max_textures", i32 128}
!12 = !{i32 7, !"air.max_read_write_textures", i32 8}
!13 = !{i32 7, !"air.max_samplers", i32 16}
!14 = !{!"air.compile.denorms_disable"}
!15 = !{!"air.compile.fast_math_enable"}
!16 = !{!"air.compile.framebuffer_fetch_enable"}
!17 = !{!"air.visible_function_reference", float (float)* @transform_value.MTL_VISIBLE_FN_REF, !"transform_value"}
!18 = !{!"Apple metal version 32023.850 (metalfe-32023.850)"}
!19 = !{i32 2, i32 8, i32 0}
!20 = !{!"Metal", i32 4, i32 0, i32 0}
!21 = !{!"reverse_kernel.ll"}
