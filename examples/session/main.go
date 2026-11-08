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

	// Persistent session with memory across turns
	fmt.Println("Running persistent session with memory...")
	fmt.Println("---------------------------------------------")
	s, err := engine.NewSession(goov.ChatOpts{
		MaxTokens:    25,
		Temperature:  0,
		SystemPrompt: "You remember cities I have visited.",
		MaxHistory:   10,
	})
	if err != nil {
		fmt.Println("Failed to create session:", err)
		os.Exit(1)
	}
	defer s.Close()

	prompts := []string{
		"I have been to Paris.",
		"I went to New York last year.",
		"I remember the time I lived in Lima.",
		"What places I have been to?",
	}

	fmt.Println("Prompt (System):", s.Opts.SystemPrompt)
	for _, p := range prompts {
		fmt.Println("Prompt (User):", p)
		start := time.Now()
		resp, err := s.Generate(p)
		elapsed := time.Since(start)
		if err != nil {
			fmt.Println("Failed to generate response:", err)
			os.Exit(1)
		}
		fmt.Printf("Response: %s\n", resp)
		fmt.Printf("\n >> Generation time: %s\n\n", elapsed)
	}
}
