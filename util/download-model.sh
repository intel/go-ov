#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: MIT

# Use this script to download a model from HuggingFace
# Usage:
#   sh ./download-model.sh OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov

set -e

HF_REPO_ID=$1
MODEL_DIR="../models/$1"

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -eq 0 ]; then
    echo 'Missing model to download.'
    echo 'Usage: ./download-model.sh OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov'
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
	pip install huggingface_hub
	python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="$HF_REPO_ID", local_dir="$MODEL_DIR")
EOF
fi

echo Done!
