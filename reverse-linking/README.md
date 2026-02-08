# Reverse Linking: JIT Async Copy on Metal (Xcode 26.2+)

Xcode 26.2 killed `__asm("air.simdgroup_async_copy...")` — the only way Metal
projects like [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
could use simdgroup async copy via JIT compilation.

This directory contains a **working solution** that achieves all four constraints:

| Constraint | Status |
|---|---|
| JIT (dynamic kernel generation) | YES |
| Real async DMA (hardware, not manual SIMD) | YES |
| No Xcode/xcrun at runtime | YES |
| App Store safe (all public APIs) | YES |

## How It Works

**Key insight**: `air.simdgroup_async_copy_1d.p3i8.p1i8` only works when called
from the kernel entry point body — it becomes a NO-OP when called from linked
visible functions. So we flip the architecture:

1. **Pre-compiled kernel shell** (built once with `xcrun`, shipped in app):
   Contains the kernel entry point with async copy intrinsics + a call to an
   externally-defined `[[visible]]` function.

2. **JIT-compiled visible function** (`makeLibrary(source:)` at runtime):
   Contains all the dynamic math — GEMM, softmax, attention, anything.

3. **Linked at pipeline creation** via `MTLLinkedFunctions.privateFunctions`:
   Stitches the pre-compiled kernel shell with the JIT-compiled math function.

```
BUILD TIME                          RUNTIME
┌──────────────────────┐           ┌──────────────────────┐
│  LLVM IR with        │           │  Metal source with   │
│  async copy +        │  ──────>  │  [[visible]] func    │
│  visible fn call     │  embed    │  (any dynamic math)  │
│  ↓                   │  .metallib│  ↓                   │
│  xcrun metal -cc1    │           │  makeLibrary(source:)│
│  xcrun metallib      │           │                      │
└──────────────────────┘           └──────────────────────┘
         │                                    │
         └──────────┐    ┌────────────────────┘
                    ▼    ▼
            privateFunctions linking
                    │
                    ▼
              GPU execution
         (async DMA + dynamic math)
```

## Files

- `simple_kernel.ll` — Minimal example: 1 buffer, 1D async copy, scalar visible fn
- `multi_buffer_kernel.ll` — Realistic: 2 buffers (A, B), 2x 1D async copies, visible fn gets threadgroup ptrs
- `copy_2d_kernel.ll` — 2D async copy roundtrip: strided device↔threadgroup (8x4 tile in 8x16 matrix)
- `*.metallib` — Pre-compiled from the .ll files (rebuild with `./build.sh`)
- `test_simple.swift` — Tests simple kernel with 1 dynamic function (128 elements)
- `test_multi_buffer.swift` — Tests multi-buffer kernel with 3 different dynamic functions (1024 elements)
- `test_2d_copy.swift` — Tests 2D async copy roundtrip: strided read, transform, strided write (8x4 tile)
- `build.sh` — Rebuilds metallibs from IR sources

## Quick Start

```bash
# Build metallibs (one-time, requires Xcode)
./build.sh

# Run tests (no Xcode needed — just Metal runtime)
swiftc -framework Metal -framework Foundation -o test_simple test_simple.swift && ./test_simple
swiftc -framework Metal -framework Foundation -o test_multi test_multi_buffer.swift && ./test_multi
swiftc -framework Metal -framework Foundation -o test_2d test_2d_copy.swift && ./test_2d
```

## IR Patterns

### Calling a visible function from IR
```llvm
; Declaration (externally defined, linked at pipeline time)
declare void @compute_tile.MTL_VISIBLE_FN_REF(float addrspace(3)*, ...)
    local_unnamed_addr section "air.externally_defined"

; Metadata linking the reference to the visible function name
!air.visible_function_references = !{!N}
!N = !{!"air.visible_function_reference",
       void (float addrspace(3)*, ...)* @compute_tile.MTL_VISIBLE_FN_REF,
       !"compute_tile"}
```

### 1D async copy intrinsic (must be in kernel body)
```llvm
; Device -> Threadgroup
declare %event_t addrspace(3)* @air.simdgroup_async_copy_1d.p3i8.p1i8(
    i64,                 ; sizeof(T) — e.g. 4 for float
    i64,                 ; num_elements
    i8 addrspace(3)*,    ; threadgroup destination
    i8 addrspace(1)*,    ; device source
    i64                  ; stride (0 = contiguous)
)
; Threadgroup -> Device (swap p1/p3 in name AND pointer types)
declare %event_t addrspace(3)* @air.simdgroup_async_copy_1d.p1i8.p3i8(
    i64, i64, i8 addrspace(1)*, i8 addrspace(3)*, i64)

declare void @air.wait_simdgroup_events(i32, %event_t addrspace(3)**)
```

### 2D async copy intrinsic (must be in kernel body)
```llvm
; Device -> Threadgroup (strided 2D tile copy)
declare %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
    i64,                 ; sizeof(T)
    i64,                 ; alignof(T)
    i8 addrspace(3)*,    ; threadgroup destination
    i64,                 ; dst elements per row
    i64,                 ; dst leading dimension scale (usually 1)
    <2 x i64>,           ; dst tile dims <width, height>
    i8 addrspace(1)*,    ; device source
    i64,                 ; src elements per row
    i64,                 ; src leading dimension scale (usually 1)
    <2 x i64>,           ; src tile dims <width, height>
    <2 x i64>,           ; offset <col, row> (usually <0,0>)
    i32                  ; clamp mode (0 = none)
)
; Threadgroup -> Device (swap p1/p3 in name AND pointer types)
declare %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p1i8.p3i8(
    i64, i64,
    i8 addrspace(1)*, i64, i64, <2 x i64>,
    i8 addrspace(3)*, i64, i64, <2 x i64>,
    <2 x i64>, i32)
```

### Gotcha: threadgroup globals
Use ONE global with offsets — separate globals may alias:
```llvm
; GOOD: single buffer with GEP offsets
@tg_buf = internal addrspace(3) global [64 x float] undef, align 4
%A_tg = getelementptr [64 x float], [64 x float] addrspace(3)* @tg_buf, i64 0, i64 0
%B_tg = getelementptr [64 x float], [64 x float] addrspace(3)* @tg_buf, i64 0, i64 32

; BAD: separate globals (may share same memory)
@tg_A = internal addrspace(3) global [32 x float] undef
@tg_B = internal addrspace(3) global [32 x float] undef
```

## Tested On

- macOS 26.3, Apple M1 Pro
- Xcode 26.2 (Metal toolchain 32023.850)
