// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

package goov

import (
	"bufio"
	"io"
	"os/exec"
)

// OV_INFERENCE_DEVICE identifies the OpenVINO device used to run inference:
//   - OV_INFERENCE_DEVICE_AUTO: let OpenVINO pick the best available device (with CPU fallback)
//   - OV_INFERENCE_DEVICE_CPU:  run on the CPU
//   - OV_INFERENCE_DEVICE_GPU:  run on the Intel integrated or discrete GPU
//   - OV_INFERENCE_DEVICE_NPU:  run on the Intel NPU (AI Boost)
type OV_INFERENCE_DEVICE int

const (
	OV_INFERENCE_DEVICE_AUTO OV_INFERENCE_DEVICE = iota
	OV_INFERENCE_DEVICE_CPU
	OV_INFERENCE_DEVICE_GPU
	OV_INFERENCE_DEVICE_NPU
)

// String returns the device identifier expected by the OpenVINO runtime.
func (d OV_INFERENCE_DEVICE) String() string {
	switch d {
	case OV_INFERENCE_DEVICE_AUTO:
		return "AUTO"
	case OV_INFERENCE_DEVICE_CPU:
		return "CPU"
	case OV_INFERENCE_DEVICE_GPU:
		return "GPU"
	case OV_INFERENCE_DEVICE_NPU:
		return "NPU"
	default:
		return ""
	}
}

// OV_PERFORMANCE_MODE selects the OpenVINO performance hint:
//   - OV_PERFORMANCE_MODE_UNDEFINED:  no hint; use the OpenVINO defaults
//   - OV_PERFORMANCE_MODE_LATENCY:    optimize for low latency / fast first tokens
//   - OV_PERFORMANCE_MODE_THROUGHPUT: optimize for total throughput across requests
type OV_PERFORMANCE_MODE int

const (
	OV_PERFORMANCE_MODE_UNDEFINED OV_PERFORMANCE_MODE = iota
	OV_PERFORMANCE_MODE_LATENCY
	OV_PERFORMANCE_MODE_THROUGHPUT
)

// String returns the performance hint expected by the OpenVINO runtime.
func (m OV_PERFORMANCE_MODE) String() string {
	switch m {
	case OV_PERFORMANCE_MODE_UNDEFINED:
		return "UNDEFINED"
	case OV_PERFORMANCE_MODE_LATENCY:
		return "LATENCY"
	case OV_PERFORMANCE_MODE_THROUGHPUT:
		return "THROUGHPUT"
	default:
		return ""
	}
}

type EngineOpts struct {
	ModelPath       string              // Directory where to find the LLM files
	LibraryPath     string              // Directory where to find the LLM library files
	InferenceDevice OV_INFERENCE_DEVICE // Device where to run the inference (AUTO, CPU, GPU, NPU)
	CPUPinning      bool                // enable/disable CPU pinning
	PerformanceMode OV_PERFORMANCE_MODE // LATENCY, THROUGHPUT, UNDEFINED
	NumThreads      int                 // Number of threads to use (0,1,2,4)
	DebugMode       bool                // Set to true to show additional debug output
}

type ChatOpts struct {
	MaxTokens    uint    // Maximum number of tokens to generate on responses. Use 0 to let OpenVINO decide
	Temperature  float32 // Value for the temperature to use on responses
	SystemPrompt string  // Optional system prompt applied via the chat template
	MaxHistory   int     // Maximum number of retained user/assistant messages (excludes system)
}

type LLMEngine struct {
	Opts       EngineOpts
	workerPath string // Path to the OpenVINO worker executable
}

type LLMSession struct {
	Opts   ChatOpts
	cmd    *exec.Cmd      // the running OpenVINO worker process for this session
	stdin  io.WriteCloser // prompts are written here, each terminated by an <EOL> line
	stdout *bufio.Reader  // worker responses are read from here until an <EOL> line
}

type LLMBatch struct {
	Opts   ChatOpts
	cmd    *exec.Cmd      // the running OpenVINO worker process for this batch session
	stdin  io.WriteCloser // prompts are written here, each terminated by an <EOL> line
	stdout *bufio.Reader  // worker responses are read from here until an <EOL> line
}

// Chunk is a single piece of streamed output: Text carries generated text as it is
// produced, or Err is set if streaming failed.
type Chunk struct {
	Text string
	Err  error
}
