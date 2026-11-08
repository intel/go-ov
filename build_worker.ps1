# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: MIT

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found at $vswhere" }

$vsInstall = & $vswhere -latest -products * -property installationPath
if (-not $vsInstall) { throw "No Visual Studio installation found" }

$vcvars = Join-Path $vsInstall 'VC\Auxiliary\Build\vcvars64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

# Import the MSVC x64 environment produced by vcvars64.bat into this session.
cmd /c "`"$vcvars`" && set" | ForEach-Object {
    if ($_ -match '^(.*?)=(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] }
}

$ovRuntime = 'vendor\openvino_genai_windows_2026.4.0.0_x86_64\runtime'
$include = Join-Path $ovRuntime 'include'
$libDir = Join-Path $ovRuntime 'lib\intel64\Release'

# /W4 /WX hold our code to warnings-as-errors; external: flags mute third-party/system headers.
& cl /nologo /std:c++17 /EHsc /O2 /sdl /guard:cf /W4 /WX /external:W0 /external:env:INCLUDE "/external:I$include" worker\ov_worker.cpp /Fe:ov_worker.exe /Fo:ov_worker.obj /link "/LIBPATH:$libDir" /guard:cf /CETCOMPAT openvino_genai.lib openvino.lib
if ($LASTEXITCODE -ne 0) { throw "cl.exe failed with exit code $LASTEXITCODE" }

Remove-Item -ErrorAction SilentlyContinue ov_worker.obj
Write-Host '[build_worker] ov_worker.exe built'
