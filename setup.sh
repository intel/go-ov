#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: MIT

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

HF_REPO_ID="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"
OV_VERSION="2026.4.1.0"
OV_CHANNEL="2026.4"

MODEL_DIR="models/qwen2.5-1.5b-int4"
OV_DIR="vendor/openvino_genai/${OV_VERSION}_x86_64"
OV_PKG="openvino_genai_windows_${OV_VERSION}_x86_64"
OV_URL="${OPENVINO_GENAI_URL:-https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/${OV_CHANNEL}/windows/${OV_PKG}.zip}"


# Download and extract OpenVINO GenAI runtime
if [ ! -d "$OV_DIR" ]; then
	curl -L --fail -o "$OV_PKG.zip" "$OV_URL"
	powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
		"Expand-Archive -Path '$OV_PKG.zip' -DestinationPath 'vendor/openvino_genai' -Force"
	mv "vendor/openvino_genai/$OV_PKG" "$OV_DIR"
	rm -f "$OV_PKG.zip"
fi


# Download model
if [ ! -d "$MODEL_DIR" ]; then
	pip install huggingface_hub
	python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$HF_REPO_ID", local_dir="$MODEL_DIR")
EOF
fi
