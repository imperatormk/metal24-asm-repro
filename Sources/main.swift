// metal24-asm-repro: Prove that -std=macos-metal2.4 enables __asm directives
// on macOS 26 (Tahoe) for JIT / codegen-style Metal kernel compilation.
//
// The shader source is assembled from Swift string interpolation at runtime,
// exactly like metal-flash-attention's createSource() pattern -- NOT read
// from a .metal file.
//
// Usage: swift run

import Foundation
import Metal

// ---------------------------------------------------------------------------
// MARK: - Helpers

// ---------------------------------------------------------------------------

func shell(_ command: String) -> (output: String, status: Int32) {
    let task = Process()
    let pipe = Pipe()
    task.standardOutput = pipe
    task.standardError = pipe
    task.launchPath = "/bin/bash"
    task.arguments = ["-c", command]
    do {
        try task.run()
        task.waitUntilExit()
    } catch {
        return ("Failed to launch: \(error)", -1)
    }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    let output = String(data: data, encoding: .utf8) ?? ""
    return (output, task.terminationStatus)
}

func separator(_ title: String) {
    let line = String(repeating: "=", count: 72)
    print("\n\(line)")
    print("  \(title)")
    print(line)
}

// ---------------------------------------------------------------------------
// MARK: - Codegen: build Metal source from Swift string interpolation

// ---------------------------------------------------------------------------

// These functions mirror metal-flash-attention's codegen pattern:
//   AttentionKernel.createSource()
//     -> createMetalSimdgroupEvent()   // __asm declarations
//     -> createBufferBindings()         // device buffer args
//     -> loopForward() etc.             // kernel body
//
// The multiline strings use the same column-0 content style as MFA.

// swiftformat:disable all
func createSimdgroupEventHeader() -> String {
"""
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
"""
}

func createBufferBindings(elementType: String) -> String {
"""
  device const \(elementType) *src [[buffer(0)]],
  device \(elementType)       *dst [[buffer(1)]],
"""
}

func createAsyncCopyBody(elements: Int) -> String {
"""
  threadgroup float tg_buf[\(elements)];

  // async copy \(elements) floats: device -> threadgroup
  thread _simdgroup_event_t* ev =
      __metal_simdgroup_async_copy_1d(
          sizeof(float),
          \(elements),
          (threadgroup void*)tg_buf,
          (const device void*)(src + (tid / \(elements)) * \(elements)),
          0);

  __metal_wait_simdgroup_events(1, &ev);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  dst[tid] = tg_buf[simd_lane] * 2.0f;
"""
}

func createSource(elements: Int, elementType: String) -> String {
"""
#include <metal_stdlib>
using namespace metal;

\(createSimdgroupEventHeader())

kernel void async_copy_test(
  \(createBufferBindings(elementType: elementType))
  uint tid [[thread_position_in_grid]],
  uint simd_lane [[thread_index_in_simdgroup]])
{
\(createAsyncCopyBody(elements: elements))
}
"""
}
// swiftformat:enable all

// ---------------------------------------------------------------------------
// MARK: - Compile via CLI (codegen path)

// ---------------------------------------------------------------------------

/// Compiles codegen'd source via CLI with -std=macos-metal2.4, matching
/// MTLLibraryCompiler.makeLibraryViaCLI().
func compileViaCLI(source: String, device: MTLDevice) throws -> MTLLibrary {
    let fm = FileManager.default
    let tmp = fm.temporaryDirectory
    let id = UUID().uuidString

    let srcURL = tmp.appendingPathComponent("mfa_\(id).metal")
    let airURL = tmp.appendingPathComponent("mfa_\(id).air")
    let libURL = tmp.appendingPathComponent("mfa_\(id).metallib")

    defer {
        try? fm.removeItem(at: srcURL)
        try? fm.removeItem(at: airURL)
        try? fm.removeItem(at: libURL)
    }

    try source.write(to: srcURL, atomically: true, encoding: .utf8)

    let compile = shell(
        "xcrun -sdk macosx metal -std=macos-metal2.4 -c '\(srcURL.path)' -o '\(airURL.path)' 2>&1"
    )
    guard compile.status == 0 else {
        throw NSError(
            domain: "repro", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "compile: \(compile.output)"]
        )
    }

    let link = shell(
        "xcrun -sdk macosx metallib '\(airURL.path)' -o '\(libURL.path)' 2>&1"
    )
    guard link.status == 0 else {
        throw NSError(
            domain: "repro", code: 2,
            userInfo: [NSLocalizedDescriptionKey: "link: \(link.output)"]
        )
    }

    return try device.makeLibrary(URL: libURL)
}

// ---------------------------------------------------------------------------
// MARK: - Main

// ---------------------------------------------------------------------------

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No Metal device found")
}

let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

var report = ""

func log(_ msg: String) {
    print(msg)
    report += msg + "\n"
}

separator("metal24-asm-repro (codegen style)")
log("Date        : \(Date())")
log("macOS       : \(ProcessInfo.processInfo.operatingSystemVersionString)")
log("Device      : \(device.name)")
log("Metal       : \(shell("xcrun metal --version 2>&1").output.trimmingCharacters(in: .whitespacesAndNewlines))")
log("")

// Generate shader source from Swift string interpolation (codegen)
let elements = 32
let shaderSource = createSource(elements: elements, elementType: "float")

log("--- Generated shader source (\(shaderSource.count) chars) ---")
log(shaderSource)
log("--- End generated source ---")
log("")

// ---------------------------------------------------------------------------
// MARK: - Test 1: Runtime compilation (should FAIL on macOS 26)

// ---------------------------------------------------------------------------

separator("Test 1: Runtime makeLibrary(source:) with codegen'd shader")

do {
    _ = try device.makeLibrary(source: shaderSource, options: nil)
    log("RESULT: PASS (runtime accepted __asm -- unexpected on macOS 26!)")
} catch {
    let snippet = String(String(describing: error).prefix(200))
    log("RESULT: FAIL (expected on macOS 26)")
    log("Error : \(snippet)")
}

// ---------------------------------------------------------------------------
// MARK: - Test 2: CLI compile with -std=macos-metal2.4 (should SUCCEED)

// ---------------------------------------------------------------------------

separator("Test 2: CLI xcrun metal -std=macos-metal2.4 with codegen'd shader")

var library: MTLLibrary?
do {
    library = try compileViaCLI(source: shaderSource, device: device)
    log("RESULT: PASS -- codegen + CLI + metal2.4 works on macOS 26!")
} catch {
    log("RESULT: FAIL -- \(error)")
}

// ---------------------------------------------------------------------------
// MARK: - Test 3: GPU execution

// ---------------------------------------------------------------------------

separator("Test 3: GPU execution of codegen'd kernel")

var gpuOK = false
if let lib = library {
    do {
        guard let function = lib.makeFunction(name: "async_copy_test") else {
            throw NSError(
                domain: "repro", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "function not found"]
            )
        }
        let pipeline = try device.makeComputePipelineState(function: function)

        let N = elements
        let srcBuf = device.makeBuffer(
            length: N * MemoryLayout<Float>.size, options: .storageModeShared
        )!
        let dstBuf = device.makeBuffer(
            length: N * MemoryLayout<Float>.size, options: .storageModeShared
        )!

        let srcPtr = srcBuf.contents().bindMemory(to: Float.self, capacity: N)
        for i in 0 ..< N {
            srcPtr[i] = Float(i + 1)
        }

        let queue = device.makeCommandQueue()!
        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
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

        if cmdBuf.status == .completed {
            let dstPtr = dstBuf.contents().bindMemory(to: Float.self, capacity: N)
            var allCorrect = true
            var sample = [String]()
            for i in 0 ..< N {
                let expected = Float(i + 1) * 2.0
                if abs(dstPtr[i] - expected) > 0.001 {
                    allCorrect = false
                    log("MISMATCH at [\(i)]: got \(dstPtr[i]), expected \(expected)")
                }
                if i < 4 || i == N - 1 {
                    sample.append("[\(i)]=\(dstPtr[i])")
                }
            }
            log("Sample : \(sample.joined(separator: ", "))")
            if allCorrect {
                log("RESULT : PASS -- kernel ran correctly! All \(N) values match.")
                gpuOK = true
            } else {
                log("RESULT : FAIL -- values mismatch")
            }
        } else {
            log("RESULT : FAIL -- command buffer status: \(cmdBuf.status.rawValue)")
        }
    } catch {
        log("RESULT : FAIL -- \(error)")
    }
}

// ---------------------------------------------------------------------------
// MARK: - Summary

// ---------------------------------------------------------------------------

separator("Summary")
log("""
Shader source was assembled at runtime via Swift string interpolation,
mirroring metal-flash-attention's codegen pattern (createSource()):

  createSource()
    -> createSimdgroupEventHeader()   // __asm declarations
    -> createBufferBindings()          // device buffer args
    -> createAsyncCopyBody()           // kernel body

Then compiled via CLI:
  xcrun metal -std=macos-metal2.4 -c src.metal -o src.air
  xcrun metallib src.air -o src.metallib
  device.makeLibrary(URL: metallib)

This proves the codegen->CLI->metallib->GPU path works on macOS 26.
""")

if gpuOK {
    log("ALL TESTS PASSED")
} else {
    log("SOME TESTS FAILED -- see details above")
}

let reportPath = cwd.appendingPathComponent("results.txt")
try! report.write(to: reportPath, atomically: true, encoding: .utf8)
log("\nReport written to: \(reportPath.path)")
