/// Multi-buffer reverse linking test:
/// Pre-compiled kernel async-copies A and B tiles into threadgroup,
/// then calls a JIT-compiled [[visible]] function for the math.
/// Tests 3 different dynamic functions, 1024 elements each.
import Metal
import Foundation

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("No Metal device") }
print("Device: \(device.name)")

let scriptDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let metallibData = try Data(contentsOf: URL(fileURLWithPath: "\(scriptDir)/multi_buffer_kernel.metallib"))
let metallibDispatch = metallibData.withUnsafeBytes { DispatchData(bytes: $0) }
let precompiledLib = try device.makeLibrary(data: metallibDispatch)

let sources: [(String, String, (Float, Float) -> Float)] = [
    ("A*B+42", """
    #include <metal_stdlib>
    using namespace metal;
    [[visible]]
    void compute_tile(threadgroup float* A, threadgroup float* B, device float* C, uint tid, uint lane) {
        C[tid] = A[lane] * B[lane] + 42.0f;
    }
    """, { a, b in a * b + 42.0 }),

    ("fma(A,B,100)", """
    #include <metal_stdlib>
    using namespace metal;
    [[visible]]
    void compute_tile(threadgroup float* A, threadgroup float* B, device float* C, uint tid, uint lane) {
        C[tid] = fma(A[lane], B[lane], 100.0f);
    }
    """, { a, b in a * b + 100.0 }),

    ("sqrt(A)+B*2", """
    #include <metal_stdlib>
    using namespace metal;
    [[visible]]
    void compute_tile(threadgroup float* A, threadgroup float* B, device float* C, uint tid, uint lane) {
        C[tid] = sqrt(A[lane]) + B[lane] * 2.0f;
    }
    """, { a, b in sqrt(a) + b * 2.0 }),
]

let N = 1024
let A_data = (0..<N).map { Float($0) + 1.0 }
let B_data = (0..<N).map { Float($0) * 0.5 + 0.1 }
let A_buf = device.makeBuffer(bytes: A_data, length: N * 4, options: .storageModeShared)!
let B_buf = device.makeBuffer(bytes: B_data, length: N * 4, options: .storageModeShared)!
let queue = device.makeCommandQueue()!
guard let kernelFn = precompiledLib.makeFunction(name: "gemm_kernel") else { fatalError() }

var allPassed = true
for (name, source, expected_fn) in sources {
    let jitLib = try device.makeLibrary(source: source, options: nil)
    guard let visibleFn = jitLib.makeFunction(name: "compute_tile") else { fatalError() }

    let linkedFns = MTLLinkedFunctions()
    linkedFns.privateFunctions = [visibleFn]
    let desc = MTLComputePipelineDescriptor()
    desc.computeFunction = kernelFn
    desc.linkedFunctions = linkedFns

    let pipeline = try device.makeComputePipelineState(descriptor: desc, options: [], reflection: nil)
    let C_buf = device.makeBuffer(length: N * 4, options: .storageModeShared)!

    let cmdBuf = queue.makeCommandBuffer()!
    let encoder = cmdBuf.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(A_buf, offset: 0, index: 0)
    encoder.setBuffer(B_buf, offset: 0, index: 1)
    encoder.setBuffer(C_buf, offset: 0, index: 2)
    encoder.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    encoder.endEncoding()
    cmdBuf.commit()
    cmdBuf.waitUntilCompleted()

    let result = C_buf.contents().bindMemory(to: Float.self, capacity: N)
    var correct = 0
    for i in 0..<N {
        let expected = expected_fn(A_data[i], B_data[i])
        if abs(result[i] - expected) < 0.1 { correct += 1 }
    }
    let status = correct == N ? "PASS" : "FAIL"
    if correct != N { allPassed = false }
    print("[\(status)] \(name): \(correct)/\(N)")
}

print(allPassed ? "\nAll tests passed!" : "\nSome tests failed!")
