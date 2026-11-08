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

	// One-shot text generation w/o persistent session
	fmt.Println("Running one-shot text generation...")
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
	resp, err := engine.Generate(prompt, cfg)
	elapsed := time.Since(start)
	if err != nil {
		fmt.Println("Failed to generate response:", err)
		os.Exit(1)
	}
	fmt.Println("Response:", resp)
	fmt.Printf("\n >> Generation time: %s\n", elapsed)
}
