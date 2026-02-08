#!/bin/bash
# Build the pre-compiled kernel metallibs from LLVM IR sources.
# Requires Xcode with Metal toolchain (build-time only — not needed at runtime).
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Building kernel shells ==="

for name in simple_kernel multi_buffer_kernel copy_2d_kernel; do
    echo "  ${name}.ll → .air → .metallib"
    xcrun metal -cc1 -triple air64_v28-apple-macosx26.0.0 -emit-llvm-bc \
        -no-opaque-pointers -disable-llvm-verifier \
        -o "${name}.air" -x ir "${name}.ll"
    xcrun metallib "${name}.air" -o "${name}.metallib"
    rm "${name}.air"
done

echo "=== Done ==="
echo ""
echo "Run tests:"
echo "  swiftc -framework Metal -framework Foundation -o test_simple test_simple.swift && ./test_simple"
echo "  swiftc -framework Metal -framework Foundation -o test_multi test_multi_buffer.swift && ./test_multi"
echo "  swiftc -framework Metal -framework Foundation -o test_2d test_2d_copy.swift && ./test_2d"
