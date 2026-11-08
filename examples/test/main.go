// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

/*
Folders required at runtime:
- models/qwen2.5-1.5b-int4
- bin (OpenVINO DLLs, assembled by build.sh)
- resources (ov_worker.exe, assembled by build.sh)
*/

// LLMEngine drives the MSVC-built ov_worker.exe, which hosts OpenVINO in a separate process.
type LLMEngine struct {
	modelPath  string
	device     string
	workerPath string
	dllDirs    []string
}

func NewLLMEngine(modelPath string, device string) (*LLMEngine, error) {
	executablePath, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("locate executable: %w", err)
	}
	baseDir := filepath.Dir(executablePath)

	workerPath := filepath.Join(baseDir, "resources", "ov_worker.exe")
	if _, err := os.Stat(workerPath); err != nil {
		return nil, fmt.Errorf("locate OpenVINO worker %q: %w", workerPath, err)
	}

	dllDirs := []string{
		filepath.Join(baseDir, "bin"),
	}

	return &LLMEngine{
		modelPath:  modelPath,
		device:     device,
		workerPath: workerPath,
		dllDirs:    dllDirs,
	}, nil
}

func (e *LLMEngine) Generate(prompt string, maxTokens uint) (string, error) {
	cmd := exec.Command(e.workerPath,
		"--model", e.modelPath,
		"--device", e.device,
		"--max-new-tokens", strconv.FormatUint(uint64(maxTokens), 10),
	)
	cmd.Env = e.workerEnv()
	cmd.Stdin = strings.NewReader(prompt)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("worker failed: %w: %s", err, strings.TrimSpace(stderr.String()))
	}
	return stdout.String(), nil
}

func (e *LLMEngine) Close() {}

// GenerateBatch spawns the worker in --batch mode, sends every prompt terminated by <EOL>, then
// <EOF> to trigger processing, and returns the responses in input order.
func (e *LLMEngine) GenerateBatch(prompts []string, maxTokens uint) ([]string, error) {
	cmd := exec.Command(e.workerPath,
		"--model", e.modelPath,
		"--device", e.device,
		"--max-new-tokens", strconv.FormatUint(uint64(maxTokens), 10),
		"--batch",
		"--debug",
	)
	cmd.Env = e.workerEnv()
	cmd.Stderr = os.Stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("create worker stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("create worker stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start worker: %w", err)
	}

	var sb strings.Builder
	for _, p := range prompts {
		sb.WriteString(p)
		sb.WriteString("\n<EOL>\n")
	}
	sb.WriteString("<EOF>\n")
	if _, err := io.WriteString(stdin, sb.String()); err != nil {
		return nil, fmt.Errorf("send prompts: %w", err)
	}
	stdin.Close()

	// Each response is terminated by <EOL>; the whole batch output ends with <EOF>.
	reader := bufio.NewReader(stdout)
	var responses []string
	var lines []string
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("read worker output: %w", err)
		}
		trimmed := strings.TrimRight(line, "\r\n")
		if trimmed == "<EOF>" {
			break
		}
		if trimmed == "<EOL>" {
			responses = append(responses, strings.Join(lines, "\n"))
			lines = nil
			continue
		}
		lines = append(lines, trimmed)
	}

	if err := cmd.Wait(); err != nil {
		return nil, fmt.Errorf("worker failed: %w", err)
	}
	return responses, nil
}

// workerEnv returns the environment with the OpenVINO runtime and TBB directories on PATH.
func (e *LLMEngine) workerEnv() []string {
	newPath := strings.Join(append(append([]string{}, e.dllDirs...), os.Getenv("PATH")), string(os.PathListSeparator))
	env := os.Environ()
	for i, kv := range env {
		if strings.HasPrefix(strings.ToUpper(kv), "PATH=") {
			env[i] = "PATH=" + newPath
			return env
		}
	}
	return append(env, "PATH="+newPath)
}

func main() {
	fmt.Println("Loading OpenVINO Qwen 2.5 1.5B...")
	engine, err := NewLLMEngine("./models/qwen2.5-1.5b-int4", "CPU")
	if err != nil {
		panic(err)
	}
	defer engine.Close()

	prompts := []string{
		"Explain what a high disk queue length means in two short sentences.",
		"Explain what CPU idle time represents in two short sentences.",
		"List three reasons a laptop battery may drain faster than expected.",
	}

	fmt.Printf("Generating %d prompts in a single batch...\n", len(prompts))
	responses, err := engine.GenerateBatch(prompts, 200)
	if err != nil {
		panic(err)
	}

	for i, r := range responses {
		fmt.Printf("\n=== Response %d ===\n%s\n", i+1, r)
	}
}
