/// 2D async copy roundtrip test:
/// 1. Load 8x4 tile from 8x16 source matrix (strided 2D read)
/// 2. JIT visible function transforms tile in-place (*10)
/// 3. Store 8x4 tile back to 8x16 dest matrix (strided 2D write)
/// Tests air.simdgroup_async_copy_2d in both directions.
import Metal
import Foundation

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
print("Device: \(device.name)")

let scriptDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let metallibData = try Data(contentsOf: URL(fileURLWithPath: "\(scriptDir)/copy_2d_kernel.metallib"))
let metallibDispatch = metallibData.withUnsafeBytes { DispatchData(bytes: $0) }
let precompiledLib = try device.makeLibrary(data: metallibDispatch)

// Visible fn: multiply each element in threadgroup tile by 10
let visibleSource = """
#include <metal_stdlib>
using namespace metal;

[[visible]]
void process_tile(threadgroup float* tg, uint simd_lane) {
    if (simd_lane < 32) {
        tg[simd_lane] *= 10.0f;
    }
}
"""

let jitLib = try device.makeLibrary(source: visibleSource, options: nil)
guard let kernelFn = precompiledLib.makeFunction(name: "copy_2d_roundtrip_kernel"),
      let visibleFn = jitLib.makeFunction(name: "process_tile") else { fatalError() }

let linked = MTLLinkedFunctions()
linked.privateFunctions = [visibleFn]
let desc = MTLComputePipelineDescriptor()
desc.computeFunction = kernelFn
desc.linkedFunctions = linked
let pipeline = try device.makeComputePipelineState(descriptor: desc, options: [], reflection: nil)
print("Pipeline OK!")

// Source: 8x16 matrix, value = row*100 + col
let rows = 8, cols = 16
var srcData = [Float](repeating: 0, count: rows * cols)
for r in 0..<rows {
    for c in 0..<cols {
        srcData[r * cols + c] = Float(r * 100 + c)
    }
}

let srcBuf = device.makeBuffer(bytes: srcData, length: rows * cols * 4, options: .storageModeShared)!
var dstData = [Float](repeating: -1, count: rows * cols)
let dstBuf = device.makeBuffer(bytes: dstData, length: rows * cols * 4, options: .storageModeShared)!

let queue = device.makeCommandQueue()!
let cmd = queue.makeCommandBuffer()!
let enc = cmd.makeComputeCommandEncoder()!
enc.setComputePipelineState(pipeline)
enc.setBuffer(srcBuf, offset: 0, index: 0)
enc.setBuffer(dstBuf, offset: 0, index: 1)
enc.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1),
                   threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
enc.endEncoding()
cmd.commit()
cmd.waitUntilCompleted()

if let err = cmd.error {
    print("GPU error: \(err)")
} else {
    let result = dstBuf.contents().bindMemory(to: Float.self, capacity: rows * cols)
    var correct = 0

    for r in 0..<rows {
        var rowStr = "  Row \(r): "
        for c in 0..<4 {
            let idx = r * cols + c
            let expected = Float(r * 100 + c) * 10.0
            let ok = abs(result[idx] - expected) < 0.1
            if ok { correct += 1 }
            rowStr += String(format: "%6.0f", result[idx])
            if !ok { rowStr += "!" }
            rowStr += " "
        }
        if r < 3 || r == rows - 1 { print(rowStr) }
        else if r == 3 { print("  ...") }
    }

    // Verify untouched columns
    var untouched = 0
    for r in 0..<rows {
        for c in 4..<cols {
            if result[r * cols + c] == -1.0 { untouched += 1 }
        }
    }

    print("\n[2D roundtrip] Written: \(correct)/32, Preserved: \(untouched)/\(rows * (cols - 4))")
    print(correct == 32 && untouched == rows * (cols - 4) ? "[PASS]" : "[FAIL]")
}
