// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	goov "github.com/intel/go-ov"
)

func main() {
	// current working directory
	cwd, err := os.Getwd()
	if err != nil {
		fmt.Println("Failed to get current working directory:", err)
		os.Exit(1)
	}

	// Set up engine options
	opts := goov.DefaultEngineOpts()
	opts.ModelPath = filepath.Join(cwd, "models/OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov")
	opts.LibraryPath = filepath.Join(cwd, "bin")
	opts.DebugMode = false

	engine, err := goov.New(opts)
	if err != nil {
		fmt.Println("Failed to create engine:", err)
		os.Exit(1)
	}
	defer engine.Close()

	// One-shot text generation with streaming output
	fmt.Println("Running one-shot text generation (streaming)...")
	fmt.Println("---------------------------------------------")
	cfg := goov.ChatOpts{
		MaxTokens:    100,
		Temperature:  0,
		SystemPrompt: "You explain system metrics in plain English. Use only provided context.",
	}
	prompt := "CPU: 85%, Memory: 22.6/31.GB (73%)"
	fmt.Println("Prompt (System):", cfg.SystemPrompt)
	fmt.Println("Prompt (User):", prompt)

	start := time.Now()
	stream, err := engine.Stream(prompt, cfg)
	if err != nil {
		fmt.Println("Failed to start streaming:", err)
		os.Exit(1)
	}

	// Consume and drain the channel, printing tokens as they arrive.
	fmt.Print("Response: ")
	for chunk := range stream {
		if chunk.Err != nil {
			fmt.Println("\nStreaming error:", chunk.Err)
			os.Exit(1)
		}
		fmt.Print(chunk.Text)
	}
	fmt.Printf("\n\n >> Generation time: %s\n", time.Since(start))
}
