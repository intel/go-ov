#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: MIT

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

OPENVINO_ROOT="vendor/openvino_genai/2026.4.1.0_x86_64"
if [ ! -f "$OPENVINO_ROOT/runtime/include/openvino/genai/c/llm_pipeline.h" ]; then
	echo "OpenVINO GenAI headers not found under $OPENVINO_ROOT" >&2
	exit 1
fi
if [ ! -f "$OPENVINO_ROOT/runtime/lib/intel64/Release/openvino_genai_c.lib" ]; then
	echo "OpenVINO GenAI import library not found under $OPENVINO_ROOT" >&2
	exit 1
fi
if [ ! -f "$OPENVINO_ROOT/runtime/3rdparty/tbb/bin/tbb12.dll" ]; then
	echo "Bundled TBB runtime not found under $OPENVINO_ROOT/runtime/3rdparty/tbb/bin" >&2
	exit 1
fi

# Build the MSVC worker that hosts OpenVINO (ABI-compatible with the vendor runtime).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File build_worker.ps1

# Place the worker in the resources folder (to be embedded later).
mkdir -p resources
mv -f ov_worker.exe resources/

# Assemble the runtime DLLs the worker needs (CPU, GPU, NPU, AUTO) into a single root-level folder.
RUNTIME_SRC="$OPENVINO_ROOT/runtime/bin/intel64/Release"
TBB_SRC="$OPENVINO_ROOT/runtime/3rdparty/tbb/bin"
rm -rf bin
mkdir -p bin
for dll in \
	openvino.dll \
	openvino_genai.dll \
	openvino_tokenizers.dll \
	openvino_ir_frontend.dll \
	openvino_intel_cpu_plugin.dll \
	openvino_intel_gpu_plugin.dll \
	openvino_intel_npu_plugin.dll \
	openvino_intel_npu_compiler.dll \
	openvino_intel_npu_compiler_loader.dll \
	openvino_intel_npu_vm_runtime.dll \
	openvino_auto_plugin.dll; do
	cp "$RUNTIME_SRC/$dll" bin/
done
cp "$TBB_SRC/tbb12.dll" bin/

# Package the runtime DLLs into a distributable zip under dist/.
mkdir -p dist
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
	"Compress-Archive -Path 'bin/*' -DestinationPath 'dist/openvino-runtime.zip' -Force"

