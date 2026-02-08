// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "metal24-asm-repro",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "metal24-asm-repro",
            path: "Sources"
        ),
    ]
)
