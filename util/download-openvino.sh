#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: MIT

# Use this script to download the OpenVINO GenAI runtime
# Usage:
#   sh ./download-openvino.sh

set -e

OV_VERSION="2026.4.0.0"
OV_CHANNEL="2026.4"

OV_PKG="openvino_genai_windows_${OV_VERSION}_x86_64"
OV_DIR="../vendor/${OV_PKG}"
OV_URL="https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/${OV_CHANNEL}/windows/${OV_PKG}.zip"
OV_SHA256_URL="https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/${OV_CHANNEL}/windows/${OV_PKG}.zip.sha256"

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "../vendor"

# Download and extract OpenVINO GenAI runtime
if [ ! -d "$OV_DIR" ]; then
    echo "Downloading OpenVINO GenAI runtime from $OV_URL..."
    rm -rf "$OV_PKG.zip"
	curl -L --fail -o "$OV_PKG.zip" "$OV_URL"

    # get sha256 and verify
	curl -L --fail -o "$OV_PKG.zip.sha256" "$OV_SHA256_URL"
    if ! sha256sum -c "$OV_PKG.zip.sha256" --status; then
        echo "Invalid or corrupted file. Try the download again."
        exit 1
    fi

	powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
	    "\$ErrorActionPreference='Stop'; Expand-Archive -Path '$OV_PKG.zip' -DestinationPath '../vendor' -Force"

	rm -f "$OV_PKG.zip" "$OV_PKG.zip.sha256"

    echo "OpenVINO GenAI downloaded."
else
    echo "OpenVINO GenAI runtime already downloaded. To re-download, remove the directory at $OV_DIR"
fi
