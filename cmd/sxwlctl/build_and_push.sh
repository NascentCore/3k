#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Build for macOS ARM64
GOARCH=arm64 GOOS=darwin go build -o sxwlctl main.go
zip sxwlctl-darwin-arm64.zip sxwlctl
ossutil cp -f sxwlctl-darwin-arm64.zip oss://sxwl-ai/artifacts/tools/
rm sxwlctl sxwlctl-darwin-arm64.zip

# Build for macOS AMD64
GOARCH=amd64 GOOS=darwin go build -o sxwlctl main.go
zip sxwlctl-darwin-amd64.zip sxwlctl
ossutil cp -f sxwlctl-darwin-amd64.zip oss://sxwl-ai/artifacts/tools/
rm sxwlctl sxwlctl-darwin-amd64.zip

# Build for Linux AMD64
GOARCH=amd64 GOOS=linux go build -o sxwlctl main.go
zip sxwlctl-linux-amd64.zip sxwlctl
ossutil cp -f sxwlctl-linux-amd64.zip oss://sxwl-ai/artifacts/tools/
rm sxwlctl sxwlctl-linux-amd64.zip

# Build for Windows AMD64
GOARCH=amd64 GOOS=windows go build -o sxwlctl.exe main.go
zip sxwlctl-windows-amd64.zip sxwlctl.exe
ossutil cp -f sxwlctl-windows-amd64.zip oss://sxwl-ai/artifacts/tools/
rm sxwlctl.exe sxwlctl-windows-amd64.zip

echo "Build and upload completed successfully."
