/// Simple reverse linking test:
/// Pre-compiled kernel (async copy) + JIT visible function (dynamic math)
/// Result: 128/128 correct
import Metal
import Foundation

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
print("Device: \(device.name)")

// 1. Load pre-compiled metallib (contains async copy + visible function ref)
let scriptDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let metallibData = try Data(contentsOf: URL(fileURLWithPath: "\(scriptDir)/simple_kernel.metallib"))
let metallibDispatch = metallibData.withUnsafeBytes { DispatchData(bytes: $0) }
let precompiledLib = try device.makeLibrary(data: metallibDispatch)
print("Pre-compiled lib: \(precompiledLib.functionNames)")

// 2. JIT compile the visible function
let visibleSource = """
#include <metal_stdlib>
using namespace metal;

[[visible]]
float transform_value(float val) {
    return val * 3.0f + 7.0f;
}
"""
let jitLib = try device.makeLibrary(source: visibleSource, options: nil)
guard let visibleFn = jitLib.makeFunction(name: "transform_value") else { fatalError() }

// 3. Create pipeline with linked functions
guard let kernelFn = precompiledLib.makeFunction(name: "reverse_kernel") else { fatalError() }

let linkedFns = MTLLinkedFunctions()
linkedFns.privateFunctions = [visibleFn]

let desc = MTLComputePipelineDescriptor()
desc.computeFunction = kernelFn
desc.linkedFunctions = linkedFns

let pipeline = try device.makeComputePipelineState(descriptor: desc, options: [], reflection: nil)
print("Pipeline OK! maxThreads: \(pipeline.maxTotalThreadsPerThreadgroup)")

// 4. Run
let N = 128
let srcData = (0..<N).map { Float($0) + 1.0 }
let srcBuf = device.makeBuffer(bytes: srcData, length: N * 4, options: .storageModeShared)!
let dstBuf = device.makeBuffer(length: N * 4, options: .storageModeShared)!

let queue = device.makeCommandQueue()!
let cmdBuf = queue.makeCommandBuffer()!
let encoder = cmdBuf.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
encoder.setBuffer(srcBuf, offset: 0, index: 0)
encoder.setBuffer(dstBuf, offset: 0, index: 1)
encoder.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                       threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
encoder.endEncoding()
cmdBuf.commit()
cmdBuf.waitUntilCompleted()

let result = dstBuf.contents().bindMemory(to: Float.self, capacity: N)
var correct = 0
for i in 0..<N {
    let expected = (Float(i) + 1.0) * 3.0 + 7.0
    if abs(result[i] - expected) < 0.01 { correct += 1 }
}
print("Results: \(correct)/\(N) correct")
if correct == N {
    print("SUCCESS: Reverse linking works!")
}
