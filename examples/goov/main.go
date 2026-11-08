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
	//opts.ModelPath = filepath.Join(cwd, "models/qwen2.5-1.5b-int4")
	opts.ModelPath = filepath.Join(cwd, "models/qwen3.5-2b-int4")
	opts.LibraryPath = filepath.Join(cwd, "bin")
	opts.DebugMode = false

	engine, err := goov.New(opts)
	if err != nil {
		fmt.Println("Failed to create engine:", err)
		os.Exit(1)
	}
	defer engine.Close()

	// Example 1: One-shot text generation w/o persistent session
	fmt.Println("Example 1: Running one-shot text generation...")
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
	fmt.Printf("Generation time: %s\n", elapsed)

	// Example 2: Persistent session with memory
	fmt.Println("\nExample 2: Running persistent session with memory...")
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
		fmt.Printf("Generation time: %s\n\n", elapsed)
	}

	// Example 3: Batch generation
	// fmt.Println("\nExample 3: Running batch generation...")
	// fmt.Println("---------------------------------------------")
	// b, err := engine.NewBatch(goov.ChatOpts{
	// 	MaxTokens:    50,
	// 	Temperature:  0.7,
	// 	SystemPrompt: "You summarize the provided information.",
	// })
	// if err != nil {
	// 	fmt.Println("Failed to create batch session:", err)
	// 	os.Exit(1)
	// }
	// defer b.Close()

	// bPrompts := []string{
	// 	"The sky is blue.",
	// 	"The sky is big.",
	// 	"The sky sometimes has clouds.",
	// }

	// for _, p := range bPrompts {
	// 	fmt.Println("Prompt (User):", p)
	// 	if err := b.AddPrompt(p); err != nil {
	// 		fmt.Println("Failed to add prompt to batch:", err)
	// 		os.Exit(1)
	// 	}
	// }

	// start = time.Now()
	// bResponses, err := b.Generate()
	// elapsed = time.Since(start)
	// if err != nil {
	// 	fmt.Println("Failed to generate batch responses:", err)
	// 	os.Exit(1)
	// }
	// for i, resp := range bResponses {
	// 	fmt.Printf("Response %d: %s\n", i+1, resp)
	// }
	// fmt.Printf("Batch generation time: %s\n", elapsed)

}
