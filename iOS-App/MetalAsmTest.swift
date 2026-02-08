// MetalAsmTest — iOS app to test async copy JIT compilation on device
//
// Tests two approaches:
// 1. Old __asm("air.simdgroup_async_copy...") pattern (worked in Xcode 16, killed in Xcode 26)
// 2. New __metal_async_wg_copy builtin (Xcode 26.2+, no __asm needed!)
//
// On macOS 26.3 the new builtin JIT-compiles but pipeline creation fails
// ("unlowered function call"). Does iOS fare better?

import SwiftUI
import Metal

// MARK: - Shader sources

// OLD approach: __asm declarations (Xcode 16 era)
let asmSource = """
#include <metal_stdlib>
using namespace metal;

struct _simdgroup_event_t;

thread _simdgroup_event_t*
__metal_simdgroup_async_copy_1d(
  ulong, ulong, threadgroup void *, const device void *, ulong)
  __asm("air.simdgroup_async_copy_1d.p3i8.p1i8");

thread _simdgroup_event_t*
__metal_simdgroup_async_copy_1d(
  ulong, ulong, device void *, const threadgroup void *, ulong)
  __asm("air.simdgroup_async_copy_1d.p1i8.p3i8");

void __metal_wait_simdgroup_events(
  int, thread _simdgroup_event_t**)
  __asm("air.wait_simdgroup_events");

kernel void async_copy_test(
  device const float *src [[buffer(0)]],
  device float       *dst [[buffer(1)]],
  uint tid [[thread_position_in_grid]],
  uint simd_lane [[thread_index_in_simdgroup]])
{
  threadgroup float tg_buf[32];

  thread _simdgroup_event_t* ev =
      __metal_simdgroup_async_copy_1d(
          sizeof(float), 32,
          (threadgroup void*)tg_buf,
          (const device void*)(src + (tid / 32) * 32), 0);

  __metal_wait_simdgroup_events(1, &ev);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  dst[tid] = tg_buf[simd_lane] * 2.0f;
}
"""

// NEW approach: __metal_async_wg_copy builtin (Xcode 26.2+)
let builtinSource = """
#include <metal_stdlib>
using namespace metal;

kernel void async_copy_test(
  device const float *src [[buffer(0)]],
  device float       *dst [[buffer(1)]],
  uint tid [[thread_position_in_grid]],
  uint simd_lane [[thread_index_in_simdgroup]])
{
  threadgroup float tg_buf[32];
  __metal_threadgroup_event_t ev;

  __metal_async_wg_copy(
      (threadgroup float*)tg_buf,
      (const device float*)(src + (tid / 32) * 32),
      32ul,
      ev);

  __metal_wait_wg_events(1, &ev);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  dst[tid] = tg_buf[simd_lane] * 2.0f;
}
"""

// MARK: - Test runner

struct TestResult: Identifiable {
    let id = UUID()
    let name: String
    let passed: Bool
    let detail: String
}

func tryGPURun(device: MTLDevice, lib: MTLLibrary) -> TestResult {
    do {
        guard let fn = lib.makeFunction(name: "async_copy_test") else {
            return TestResult(name: "GPU run", passed: false, detail: "Function not found")
        }
        let pipeline = try device.makeComputePipelineState(function: fn)

        let N = 32
        guard let srcBuf = device.makeBuffer(length: N * 4, options: .storageModeShared),
              let dstBuf = device.makeBuffer(length: N * 4, options: .storageModeShared),
              let queue = device.makeCommandQueue(),
              let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            return TestResult(name: "GPU run", passed: false, detail: "Failed to create Metal objects")
        }

        let srcPtr = srcBuf.contents().bindMemory(to: Float.self, capacity: N)
        for i in 0..<N { srcPtr[i] = Float(i + 1) }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(srcBuf, offset: 0, index: 0)
        enc.setBuffer(dstBuf, offset: 0, index: 1)
        enc.dispatchThreads(
            MTLSize(width: N, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: N, height: 1, depth: 1)
        )
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if cmdBuf.status != .completed {
            return TestResult(name: "GPU run", passed: false, detail: "Command buffer status: \(cmdBuf.status.rawValue)")
        }

        let dstPtr = dstBuf.contents().bindMemory(to: Float.self, capacity: N)
        var allOK = true
        for i in 0..<N {
            if abs(dstPtr[i] - Float(i + 1) * 2.0) > 0.001 { allOK = false }
        }
        let sample = "[0]=\(dstPtr[0]), [1]=\(dstPtr[1]), [31]=\(dstPtr[31])"
        return TestResult(
            name: "GPU run",
            passed: allOK,
            detail: allOK ? "All 32 correct! \(sample)" : "MISMATCH \(sample)"
        )
    } catch {
        let msg = String(String(describing: error).prefix(250))
        return TestResult(name: "GPU pipeline", passed: false, detail: msg)
    }
}

func runTests() -> [TestResult] {
    var results = [TestResult]()

    guard let device = MTLCreateSystemDefaultDevice() else {
        results.append(TestResult(name: "Device", passed: false, detail: "No Metal device"))
        return results
    }

    results.append(TestResult(
        name: "Device",
        passed: true,
        detail: "\(device.name) — \(UIDevice.current.systemName) \(UIDevice.current.systemVersion)"
    ))

    // =========================================================================
    // Section A: Old __asm approach
    // =========================================================================

    results.append(TestResult(name: "--- OLD __asm ---", passed: true, detail: "Testing __asm(\"air.simdgroup_async_copy...\")"))

    // A1: __asm JIT with nil options
    do {
        let lib = try device.makeLibrary(source: asmSource, options: nil)
        results.append(TestResult(name: "__asm JIT (nil)", passed: true, detail: "Compiled!"))
        results.append(tryGPURun(device: device, lib: lib))
    } catch {
        results.append(TestResult(name: "__asm JIT (nil)", passed: false,
                                  detail: String(String(describing: error).prefix(200))))
    }

    // A2: __asm JIT with Metal 3.2
    do {
        let opts = MTLCompileOptions()
        opts.languageVersion = MTLLanguageVersion(rawValue: UInt((3 << 16) | 2))!
        let lib = try device.makeLibrary(source: asmSource, options: opts)
        results.append(TestResult(name: "__asm JIT (3.2)", passed: true, detail: "Compiled!"))
        results.append(tryGPURun(device: device, lib: lib))
    } catch {
        results.append(TestResult(name: "__asm JIT (3.2)", passed: false,
                                  detail: String(String(describing: error).prefix(200))))
    }

    // =========================================================================
    // Section B: New __metal_async_wg_copy builtin
    // =========================================================================

    results.append(TestResult(name: "--- NEW builtin ---", passed: true, detail: "Testing __metal_async_wg_copy (no __asm)"))

    // B1: builtin JIT with nil options
    do {
        let lib = try device.makeLibrary(source: builtinSource, options: nil)
        results.append(TestResult(name: "builtin JIT (nil)", passed: true, detail: "Compiled!"))
        results.append(tryGPURun(device: device, lib: lib))
    } catch {
        results.append(TestResult(name: "builtin JIT (nil)", passed: false,
                                  detail: String(String(describing: error).prefix(200))))
    }

    // B2: builtin JIT with Metal 3.2
    do {
        let opts = MTLCompileOptions()
        opts.languageVersion = MTLLanguageVersion(rawValue: UInt((3 << 16) | 2))!
        let lib = try device.makeLibrary(source: builtinSource, options: opts)
        results.append(TestResult(name: "builtin JIT (3.2)", passed: true, detail: "Compiled!"))
        results.append(tryGPURun(device: device, lib: lib))
    } catch {
        results.append(TestResult(name: "builtin JIT (3.2)", passed: false,
                                  detail: String(String(describing: error).prefix(200))))
    }

    // B3: builtin JIT with Metal 4.0
    if let v4 = MTLLanguageVersion(rawValue: UInt((4 << 16) | 0)) {
        do {
            let opts = MTLCompileOptions()
            opts.languageVersion = v4
            let lib = try device.makeLibrary(source: builtinSource, options: opts)
            results.append(TestResult(name: "builtin JIT (4.0)", passed: true, detail: "Compiled!"))
            results.append(tryGPURun(device: device, lib: lib))
        } catch {
            results.append(TestResult(name: "builtin JIT (4.0)", passed: false,
                                      detail: String(String(describing: error).prefix(200))))
        }
    }

    return results
}

// MARK: - UI

struct ContentView: View {
    @State private var results: [TestResult] = []
    @State private var running = false

    var body: some View {
        NavigationView {
            List {
                if results.isEmpty && !running {
                    Button("Run Async Copy JIT Tests") {
                        running = true
                        DispatchQueue.global().async {
                            let r = runTests()
                            DispatchQueue.main.async {
                                results = r
                                running = false
                            }
                        }
                    }
                    .font(.headline)
                }

                if running {
                    HStack {
                        ProgressView()
                        Text("Running...")
                    }
                }

                ForEach(results) { r in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(r.passed ? "PASS" : "FAIL")
                                .font(.caption.bold())
                                .foregroundColor(r.passed ? .green : .red)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(
                                    RoundedRectangle(cornerRadius: 4)
                                        .fill(r.passed ? Color.green.opacity(0.15) : Color.red.opacity(0.15))
                                )
                            Text(r.name)
                                .font(.headline)
                        }
                        Text(r.detail)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .lineLimit(8)
                    }
                    .padding(.vertical, 4)
                }
            }
            .navigationTitle("Metal Async Copy JIT")
        }
    }
}

@main
struct MetalAsmTestApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
