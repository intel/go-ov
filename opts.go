// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

package goov

// DefaultEngineOpts returns the default engine options.
// In this configuration, the engine is set to automatically select the inference device with default settings.
func DefaultEngineOpts() EngineOpts {
	return EngineOpts{
		InferenceDevice: OV_INFERENCE_DEVICE_AUTO,
		CPUPinning:      false,
		PerformanceMode: OV_PERFORMANCE_MODE_UNDEFINED,
		NumThreads:      0,
	}
}

// SafeEngineOpts returns engine options configured for safe CPU execution.
// In this configuration, the engine is set to use the CPU with safe settings for latency and limited threading.
func SafeEngineOpts() EngineOpts {
	return EngineOpts{
		InferenceDevice: OV_INFERENCE_DEVICE_CPU,
		CPUPinning:      false,
		PerformanceMode: OV_PERFORMANCE_MODE_LATENCY,
		NumThreads:      2,
	}
}

// DefaultSessionOpts returns the default session options.
func DefaultSessionOpts() ChatOpts {
	return ChatOpts{
		MaxTokens:   0,
		Temperature: 0,
		MaxHistory:  10,
	}
}
