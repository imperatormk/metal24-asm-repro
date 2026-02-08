# metal24-asm-repro

Minimal repro showing that Metal shader codegen with `__asm` directives (simdgroup async copy intrinsics) works on macOS 26 (Tahoe) when compiled via CLI with `-std=macos-metal2.4`.

This is the same approach used by [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) — shader source is assembled at runtime from Swift string interpolation, then compiled through `xcrun metal`.

## Run

```
swift run
```

## What it tests

| # | Test | Expected |
|---|------|----------|
| 1 | `device.makeLibrary(source:)` runtime JIT | FAIL — macOS 26 rejects `__asm` |
| 2 | `xcrun metal -std=macos-metal2.4` via CLI | PASS |
| 3 | GPU execution of the compiled kernel | PASS |

## Results

See [results.txt](results.txt) for output from an M1 Pro on macOS 26.3.
