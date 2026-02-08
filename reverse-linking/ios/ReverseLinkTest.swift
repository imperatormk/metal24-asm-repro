// ReverseLinkTest — iOS app to test reverse-linking async copy on device
//
// Tests pre-compiled kernel shells (with air.simdgroup_async_copy intrinsics)
// linked to JIT-compiled [[visible]] functions via MTLLinkedFunctions.
//
// Tests:
// 1. 1D async copy (device→threadgroup, 128 elements)
// 2. 2D async copy roundtrip (strided read + transform + strided write, 8x4 tile in 8x16 matrix)
// 3. Multi-function linking (3 different JIT visible functions with same kernel shell)

import SwiftUI
import Metal

// MARK: - Test infrastructure

struct TestResult: Identifiable {
    let id = UUID()
    let name: String
    let passed: Bool
    let detail: String
}

// MARK: - Test 1: 1D async copy (simple_kernel.metallib)

func test1D_asyncCopy(device: MTLDevice) -> [TestResult] {
    var results = [TestResult]()

    // Load pre-compiled kernel shell
    guard let url = Bundle.main.url(forResource: "simple_kernel", withExtension: "metallib") else {
        results.append(TestResult(name: "1D: load metallib", passed: false, detail: "simple_kernel.metallib not found in bundle"))
        return results
    }

    let precompiledLib: MTLLibrary
    do {
        precompiledLib = try device.makeLibrary(URL: url)
        results.append(TestResult(name: "1D: load metallib", passed: true, detail: "Functions: \(precompiledLib.functionNames)"))
    } catch {
        results.append(TestResult(name: "1D: load metallib", passed: false, detail: "\(error)"))
        return results
    }

    // JIT compile visible function
    let visibleSource = """
    #include <metal_stdlib>
    using namespace metal;
    [[visible]]
    float transform_value(float x) {
        return x * 2.0f + 1.0f;
    }
    """

    let jitLib: MTLLibrary
    do {
        jitLib = try device.makeLibrary(source: visibleSource, options: nil)
        results.append(TestResult(name: "1D: JIT compile", passed: true, detail: "Functions: \(jitLib.functionNames)"))
    } catch {
        results.append(TestResult(name: "1D: JIT compile", passed: false, detail: "\(error)"))
        return results
    }

    // Link and create pipeline
    guard let kernelFn = precompiledLib.makeFunction(name: "reverse_kernel"),
          let visibleFn = jitLib.makeFunction(name: "transform_value") else {
        results.append(TestResult(name: "1D: get functions", passed: false, detail: "Function lookup failed. Available: \(precompiledLib.functionNames)"))
        return results
    }

    let linked = MTLLinkedFunctions()
    linked.privateFunctions = [visibleFn]
    let desc = MTLComputePipelineDescriptor()
    desc.computeFunction = kernelFn
    desc.linkedFunctions = linked

    let pipeline: MTLComputePipelineState
    do {
        pipeline = try device.makeComputePipelineState(descriptor: desc, options: [], reflection: nil)
        results.append(TestResult(name: "1D: pipeline", passed: true, detail: "Created OK"))
    } catch {
        results.append(TestResult(name: "1D: pipeline", passed: false, detail: String(String(describing: error).prefix(300))))
        return results
    }

    // Execute
    let N = 128
    let srcData = (0..<N).map { Float($0) + 1.0 }
    guard let srcBuf = device.makeBuffer(bytes: srcData, length: N * 4, options: .storageModeShared),
          let dstBuf = device.makeBuffer(length: N * 4, options: .storageModeShared),
          let queue = device.makeCommandQueue(),
          let cmd = queue.makeCommandBuffer(),
          let enc = cmd.makeComputeCommandEncoder() else {
        results.append(TestResult(name: "1D: execute", passed: false, detail: "Failed to create Metal objects"))
        return results
    }

    memset(dstBuf.contents(), 0, N * 4)
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(srcBuf, offset: 0, index: 0)
    enc.setBuffer(dstBuf, offset: 0, index: 1)
    enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                       threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    if let err = cmd.error {
        results.append(TestResult(name: "1D: GPU error", passed: false, detail: "\(err)"))
        return results
    }

    let result = dstBuf.contents().bindMemory(to: Float.self, capacity: N)
    var correct = 0
    for i in 0..<N {
        let expected = srcData[i] * 2.0 + 1.0
        if abs(result[i] - expected) < 0.01 { correct += 1 }
    }
    let allZero = result[0] == 0 && result[1] == 0
    let sample = String(format: "[0]=%.1f [1]=%.1f [127]=%.1f", result[0], result[1], result[N-1])
    results.append(TestResult(
        name: "1D: results",
        passed: correct == N,
        detail: "\(correct)/\(N) correct. \(sample)\(allZero ? " (ALL ZERO — async copy was NO-OP!)" : "")"
    ))

    return results
}

// MARK: - Test 2: 2D async copy roundtrip (copy_2d_kernel.metallib)

func test2D_asyncCopy(device: MTLDevice) -> [TestResult] {
    var results = [TestResult]()

    guard let url = Bundle.main.url(forResource: "copy_2d_kernel", withExtension: "metallib") else {
        results.append(TestResult(name: "2D: load metallib", passed: false, detail: "copy_2d_kernel.metallib not found in bundle"))
        return results
    }

    let precompiledLib: MTLLibrary
    do {
        precompiledLib = try device.makeLibrary(URL: url)
        results.append(TestResult(name: "2D: load metallib", passed: true, detail: "Functions: \(precompiledLib.functionNames)"))
    } catch {
        results.append(TestResult(name: "2D: load metallib", passed: false, detail: "\(error)"))
        return results
    }

    // Visible fn: multiply each element by 10
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

    let jitLib: MTLLibrary
    do {
        jitLib = try device.makeLibrary(source: visibleSource, options: nil)
        results.append(TestResult(name: "2D: JIT compile", passed: true, detail: "OK"))
    } catch {
        results.append(TestResult(name: "2D: JIT compile", passed: false, detail: "\(error)"))
        return results
    }

    guard let kernelFn = precompiledLib.makeFunction(name: "copy_2d_roundtrip_kernel"),
          let visibleFn = jitLib.makeFunction(name: "process_tile") else {
        results.append(TestResult(name: "2D: get functions", passed: false, detail: "Function lookup failed"))
        return results
    }

    let linked = MTLLinkedFunctions()
    linked.privateFunctions = [visibleFn]
    let desc = MTLComputePipelineDescriptor()
    desc.computeFunction = kernelFn
    desc.linkedFunctions = linked

    let pipeline: MTLComputePipelineState
    do {
        pipeline = try device.makeComputePipelineState(descriptor: desc, options: [], reflection: nil)
        results.append(TestResult(name: "2D: pipeline", passed: true, detail: "Created OK"))
    } catch {
        results.append(TestResult(name: "2D: pipeline", passed: false, detail: String(String(describing: error).prefix(300))))
        return results
    }

    // Source: 8x16 matrix, value = row*100 + col
    let rows = 8, cols = 16
    var srcData = [Float](repeating: 0, count: rows * cols)
    for r in 0..<rows {
        for c in 0..<cols {
            srcData[r * cols + c] = Float(r * 100 + c)
        }
    }
    var dstData = [Float](repeating: -1, count: rows * cols)

    guard let srcBuf = device.makeBuffer(bytes: srcData, length: rows * cols * 4, options: .storageModeShared),
          let dstBuf = device.makeBuffer(bytes: dstData, length: rows * cols * 4, options: .storageModeShared),
          let queue = device.makeCommandQueue(),
          let cmd = queue.makeCommandBuffer(),
          let enc = cmd.makeComputeCommandEncoder() else {
        results.append(TestResult(name: "2D: execute", passed: false, detail: "Failed to create Metal objects"))
        return results
    }

    enc.setComputePipelineState(pipeline)
    enc.setBuffer(srcBuf, offset: 0, index: 0)
    enc.setBuffer(dstBuf, offset: 0, index: 1)
    enc.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1),
                       threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()

    if let err = cmd.error {
        results.append(TestResult(name: "2D: GPU error", passed: false, detail: "\(err)"))
        return results
    }

    let result = dstBuf.contents().bindMemory(to: Float.self, capacity: rows * cols)
    var correct = 0
    var detail = ""
    for r in 0..<rows {
        for c in 0..<4 {
            let idx = r * cols + c
            let expected = Float(r * 100 + c) * 10.0
            if abs(result[idx] - expected) < 0.1 { correct += 1 }
        }
    }
    // Check untouched columns
    var untouched = 0
    for r in 0..<rows {
        for c in 4..<cols {
            if result[r * cols + c] == -1.0 { untouched += 1 }
        }
    }

    let sample = String(format: "[0,0]=%.0f [1,0]=%.0f [7,3]=%.0f", result[0], result[cols], result[7*cols+3])
    detail = "Written: \(correct)/32, Preserved: \(untouched)/\(rows*(cols-4)). \(sample)"

    let allZero = result[0] == 0 && result[cols] == 0
    if allZero { detail += " (ALL ZERO — 2D async copy was NO-OP!)" }

    results.append(TestResult(
        name: "2D: roundtrip",
        passed: correct == 32 && untouched == rows * (cols - 4),
        detail: detail
    ))

    return results
}

// MARK: - Test 3: Multi-buffer with 3 different visible functions

func testMultiBuffer(device: MTLDevice) -> [TestResult] {
    var results = [TestResult]()

    guard let url = Bundle.main.url(forResource: "multi_buffer_kernel", withExtension: "metallib") else {
        results.append(TestResult(name: "Multi: load metallib", passed: false, detail: "multi_buffer_kernel.metallib not found in bundle"))
        return results
    }

    let precompiledLib: MTLLibrary
    do {
        precompiledLib = try device.makeLibrary(URL: url)
        results.append(TestResult(name: "Multi: load metallib", passed: true, detail: "OK"))
    } catch {
        results.append(TestResult(name: "Multi: load metallib", passed: false, detail: "\(error)"))
        return results
    }

    // Test with 3 different visible functions
    let testCases: [(String, String, (Float, Float) -> Float)] = [
        ("A*B+42", """
        #include <metal_stdlib>
        using namespace metal;
        [[visible]]
        void compute_tile(threadgroup float* A, threadgroup float* B, device float* C, uint tid, uint simd_lane) {
            C[tid] = A[simd_lane] * B[simd_lane] + 42.0f;
        }
        """, { a, b in a * b + 42.0 }),

        ("fma(A,B,100)", """
        #include <metal_stdlib>
        using namespace metal;
        [[visible]]
        void compute_tile(threadgroup float* A, threadgroup float* B, device float* C, uint tid, uint simd_lane) {
            C[tid] = metal::fma(A[simd_lane], B[simd_lane], 100.0f);
        }
        """, { a, b in a * b + 100.0 }),

        ("sqrt(A)+B*2", """
        #include <metal_stdlib>
        using namespace metal;
        [[visible]]
        void compute_tile(threadgroup float* A, threadgroup float* B, device float* C, uint tid, uint simd_lane) {
            C[tid] = metal::sqrt(A[simd_lane]) + B[simd_lane] * 2.0f;
        }
        """, { a, b in sqrtf(a) + b * 2.0 })
    ]

    let N = 1024
    var srcA = [Float](repeating: 0, count: N)
    var srcB = [Float](repeating: 0, count: N)
    for i in 0..<N {
        srcA[i] = Float(i % 32) + 1.0
        srcB[i] = Float(i % 32) + 0.5
    }

    guard let bufA = device.makeBuffer(bytes: srcA, length: N * 4, options: .storageModeShared),
          let bufB = device.makeBuffer(bytes: srcB, length: N * 4, options: .storageModeShared) else {
        results.append(TestResult(name: "Multi: buffers", passed: false, detail: "Failed"))
        return results
    }

    for (name, source, cpuFn) in testCases {
        let jitLib: MTLLibrary
        do {
            jitLib = try device.makeLibrary(source: source, options: nil)
        } catch {
            results.append(TestResult(name: "Multi: \(name)", passed: false, detail: "JIT fail: \(error)"))
            continue
        }

        guard let kernelFn = precompiledLib.makeFunction(name: "gemm_kernel"),
              let visibleFn = jitLib.makeFunction(name: "compute_tile") else {
            results.append(TestResult(name: "Multi: \(name)", passed: false, detail: "Function lookup failed"))
            continue
        }

        let linked = MTLLinkedFunctions()
        linked.privateFunctions = [visibleFn]
        let desc = MTLComputePipelineDescriptor()
        desc.computeFunction = kernelFn
        desc.linkedFunctions = linked

        do {
            let pipeline = try device.makeComputePipelineState(descriptor: desc, options: [], reflection: nil)

            guard let dstBuf = device.makeBuffer(length: N * 4, options: .storageModeShared),
                  let queue = device.makeCommandQueue(),
                  let cmd = queue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else { continue }

            memset(dstBuf.contents(), 0, N * 4)
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(bufA, offset: 0, index: 0)
            enc.setBuffer(bufB, offset: 0, index: 1)
            enc.setBuffer(dstBuf, offset: 0, index: 2)
            enc.dispatchThreads(MTLSize(width: N, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

            let result = dstBuf.contents().bindMemory(to: Float.self, capacity: N)
            var correct = 0
            for i in 0..<N {
                let expected = cpuFn(srcA[i], srcB[i])
                if abs(result[i] - expected) < 0.1 { correct += 1 }
            }
            let sample = String(format: "[0]=%.2f [1]=%.2f [2]=%.2f exp=%.2f", result[0], result[1], result[2], cpuFn(srcA[0], srcB[0]))
            let allZero = result[0] == 0 && result[1] == 0 && result[2] == 0
            results.append(TestResult(
                name: "Multi: \(name)",
                passed: correct == N,
                detail: "\(correct)/\(N) correct. \(sample)\(allZero ? " ALL ZERO — async copy NO-OP!" : "")"
            ))
        } catch {
            results.append(TestResult(name: "Multi: \(name)", passed: false, detail: "Pipeline: \(String(String(describing: error).prefix(200)))"))
        }
    }

    return results
}

// MARK: - Run all tests

func runAllTests() -> [TestResult] {
    var results = [TestResult]()

    guard let device = MTLCreateSystemDefaultDevice() else {
        results.append(TestResult(name: "Device", passed: false, detail: "No Metal device"))
        return results
    }

    results.append(TestResult(
        name: "Device",
        passed: true,
        detail: "\(device.name) — iOS \(UIDevice.current.systemVersion)"
    ))

    results.append(TestResult(name: "=== 1D Async Copy ===", passed: true, detail: "simple_kernel + JIT visible fn"))
    results.append(contentsOf: test1D_asyncCopy(device: device))

    results.append(TestResult(name: "=== 2D Roundtrip ===", passed: true, detail: "copy_2d_kernel + strided read/write"))
    results.append(contentsOf: test2D_asyncCopy(device: device))

    results.append(TestResult(name: "=== Multi-Buffer ===", passed: true, detail: "multi_buffer_kernel + 3 visible fns"))
    results.append(contentsOf: testMultiBuffer(device: device))

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
                    Button("Run Reverse Linking Tests") {
                        running = true
                        DispatchQueue.global().async {
                            let r = runAllTests()
                            DispatchQueue.main.async {
                                results = r
                                running = false
                            }
                        }
                    }
                    .font(.headline)

                    Text("Tests pre-compiled kernel shells with async copy intrinsics, linked to JIT-compiled [[visible]] functions via MTLLinkedFunctions.privateFunctions.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                if running {
                    HStack {
                        ProgressView()
                        Text("Running...")
                    }
                }

                ForEach(results) { r in
                    if r.name.hasPrefix("===") {
                        Section(header: Text(r.name).font(.caption2)) {
                            EmptyView()
                        }
                    } else {
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
                                    .font(.subheadline.bold())
                            }
                            Text(r.detail)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                                .lineLimit(8)
                        }
                        .padding(.vertical, 2)
                    }
                }
            }
            .navigationTitle("Reverse Link Test")
        }
    }
}

@main
struct ReverseLinkTestApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
