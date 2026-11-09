#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: MIT

# Use this script to assemble the OpenVINO GenAI runtime DLLs into a distributable package.
# If OpenVINO is not found, it will be downloaded automatically.
# Usage:
#   sh ./make-ov-runtime.sh

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

OV_VERSION="2026.4.1.0"
OPENVINO_ROOT="../vendor/openvino_genai_windows_${OV_VERSION}_x86_64"

if [ ! -d "$OPENVINO_ROOT" ]; then
	echo "OpenVINO not found, downloading it now."
    sh ./download-openvino.sh
fi

# Assemble the runtime DLLs the worker needs (CPU, GPU, NPU, AUTO) into a single root-level folder.
RUNTIME_SRC="$OPENVINO_ROOT/runtime/bin/intel64/Release"
TBB_SRC="$OPENVINO_ROOT/runtime/3rdparty/tbb/bin"
rm -rf ../bin
mkdir -p ../bin
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
	cp "$RUNTIME_SRC/$dll" ../bin/
done
cp "$TBB_SRC/tbb12.dll" ../bin/

# Package the runtime DLLs into a distributable zip under dist/.
mkdir -p ../dist
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
	"Compress-Archive -Path '../bin/*' -DestinationPath '../dist/openvino-runtime.zip' -Force"

echo Done!
