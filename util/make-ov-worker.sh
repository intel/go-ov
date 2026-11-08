#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: MIT

# Use this script to build the OpenVINO GenAI worker executable.
# Usage:
#   sh ./make-ov-worker.sh
    
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

OV_VERSION="2026.4.0.0"
OPENVINO_ROOT="../vendor/openvino_genai_windows_${OV_VERSION}_x86_64"

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
mkdir -p ../resources
mv -f ov_worker.exe ../resources/

echo Done!