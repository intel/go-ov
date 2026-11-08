// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

package goov

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

	"github.com/intel/go-ov/resources"
)

// OpenVINO version used by the engine
const OV_VERSION = "2026.4.1.0"
const OV_WORKER_EXE = "ov_worker.exe"

// New creates a new instance of the LLMEngine object with the given engine options.
// This function does not load the engine into memory; it only prepares the worker executable.
func New(opts EngineOpts) (*LLMEngine, error) {
	// Validate options
	if opts.ModelPath == "" {
		return nil, fmt.Errorf("ModelPath must not be empty")
	}
	if opts.LibraryPath == "" {
		return nil, fmt.Errorf("LibraryPath must not be empty")
	}
	if info, err := os.Stat(opts.ModelPath); err != nil || !info.IsDir() {
		return nil, fmt.Errorf("ModelPath %q is not an existing directory", opts.ModelPath)
	}
	if info, err := os.Stat(opts.LibraryPath); err != nil || !info.IsDir() {
		return nil, fmt.Errorf("LibraryPath %q is not an existing directory", opts.LibraryPath)
	}

	// Extract the worker into a versioned LocalAppData directory.
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return nil, fmt.Errorf("locate user cache directory: %w", err)
	}
	workerDir := filepath.Join(cacheDir, "intel", "goov", OV_VERSION)

	workerPath, err := resources.Extract(OV_WORKER_EXE, workerDir, true)
	if err != nil {
		return nil, fmt.Errorf("failed to extract OpenVINO worker: %w", err)
	}

	// Co-locate the OpenVINO runtime DLLs with the worker so they load from the worker's own
	// directory (DLL-injection hardening) rather than via PATH. Staged once per version;
	if err := resources.CopyDir(opts.LibraryPath, workerDir, false); err != nil {
		return nil, fmt.Errorf("failed to stage OpenVINO runtime: %w", err)
	}

	return &LLMEngine{
		Opts:       opts,
		workerPath: workerPath,
	}, nil
}

// Generate runs a text-generation pipeline with the given prompt and chat parameters.
// It returns the generated text or an error if the worker fails. No persistent session is created.
// Chat parameters are used to specify generation settings such as temperature, max tokens, and system prompt.
func (e *LLMEngine) Generate(prompt string, opts ChatOpts) (string, error) {
	cmd, err := e.worker(opts, "")
	if err != nil {
		return "", err
	}
	cmd.Stdin = strings.NewReader(prompt)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("worker failed: %w: %s", err, strings.TrimSpace(stderr.String()))
	}

	fmt.Println(strings.TrimSpace(stderr.String()))

	return stdout.String(), nil
}

// Stream runs a one-shot text generation and streams the output as it is produced.
// It returns a channel that emits Chunks until generation completes, after which the
// channel is closed. The caller must drain the channel to completion so the worker is
// not blocked by backpressure and its process is reaped.
func (e *LLMEngine) Stream(prompt string, opts ChatOpts) (<-chan Chunk, error) {
	cmd, err := e.worker(opts, "")
	if err != nil {
		return nil, err
	}
	cmd.Stdin = strings.NewReader(prompt)
	cmd.Stderr = os.Stderr

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("create worker stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start worker: %w", err)
	}

	// One-shot mode emits no <EOL>; the worker closes stdout (EOF) when generation ends.
	out := make(chan Chunk, 16)
	go func() {
		defer close(out)
		buf := make([]byte, 4096)
		for {
			n, rerr := stdout.Read(buf)
			if n > 0 {
				out <- Chunk{Text: string(buf[:n])}
			}
			if rerr != nil {
				if rerr != io.EOF {
					out <- Chunk{Err: fmt.Errorf("read worker output: %w", rerr)}
				}
				break
			}
		}
		if werr := cmd.Wait(); werr != nil {
			out <- Chunk{Err: fmt.Errorf("worker failed: %w", werr)}
		}
	}()

	return out, nil
}

func (e *LLMEngine) Close() error {
	// TODO: ensure we are no leaving any orphaned workers behind
	return nil
}

func (e *LLMEngine) NewSession(opts ChatOpts) (*LLMSession, error) {
	cmd, err := e.worker(opts, "interactive")
	if err != nil {
		return nil, err
	}

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
		return nil, fmt.Errorf("failed to start worker: %w", err)
	}

	return &LLMSession{
		Opts:   opts,
		cmd:    cmd,
		stdin:  stdin,
		stdout: bufio.NewReader(stdout),
	}, nil
}

func (s *LLMSession) Generate(prompt string) (string, error) {
	// Send the prompt terminated by an <EOL> line to trigger generation.
	if _, err := io.WriteString(s.stdin, prompt+"\n<EOL>\n"); err != nil {
		return "", fmt.Errorf("send prompt to worker: %w", err)
	}

	var lines []string
	for {
		line, err := s.stdout.ReadString('\n')
		if err != nil {
			return "", fmt.Errorf("read worker output: %w", err)
		}
		if strings.TrimRight(line, "\r\n") == "<EOL>" {
			break
		}
		lines = append(lines, strings.TrimRight(line, "\r\n"))
	}
	return strings.Join(lines, "\n"), nil
}

// Stream sends a prompt to the session worker and streams the response as it is produced.
// It returns a channel that emits Chunks until the <EOL> terminator is reached, after which
// the channel is closed. The caller must drain the channel before issuing the next session call.
func (s *LLMSession) Stream(prompt string) (<-chan Chunk, error) {
	// Send the prompt terminated by an <EOL> line to trigger generation.
	if _, err := io.WriteString(s.stdin, prompt+"\n<EOL>\n"); err != nil {
		return nil, fmt.Errorf("send prompt to worker: %w", err)
	}

	out := make(chan Chunk, 16)
	go func() {
		defer close(out)

		const sentinel = "\n<EOL>\n"
		var pending []byte
		buf := make([]byte, 4096)
		for {
			n, rerr := s.stdout.Read(buf)
			if n > 0 {
				pending = append(pending, buf[:n]...)
				// Stop at the terminator, emitting only the content before it.
				if idx := bytes.Index(pending, []byte(sentinel)); idx >= 0 {
					if idx > 0 {
						out <- Chunk{Text: string(pending[:idx])}
					}
					return
				}
				// Emit everything except a possible partial terminator held at the tail.
				if len(pending) > len(sentinel)-1 {
					emit := len(pending) - (len(sentinel) - 1)
					out <- Chunk{Text: string(pending[:emit])}
					pending = append(pending[:0], pending[emit:]...)
				}
			}
			if rerr != nil {
				// The interactive worker should not close before the terminator.
				out <- Chunk{Err: fmt.Errorf("read worker output: %w", rerr)}
				return
			}
		}
	}()

	return out, nil
}

func (s *LLMSession) Close() error {
	if s.cmd == nil {
		// Already closed or never started: nothing to do.
		return nil
	}
	cmd := s.cmd
	s.cmd = nil
	if _, err := io.WriteString(s.stdin, "<EOF>\n"); err != nil {
		return fmt.Errorf("send EOF to worker: %w", err)
	}
	s.stdin.Close()
	if err := cmd.Wait(); err != nil {
		return fmt.Errorf("worker failed: %w", err)
	}
	return nil
}

func (e *LLMEngine) NewBatch(opts ChatOpts) (*LLMBatch, error) {
	cmd, err := e.worker(opts, "batch")
	if err != nil {
		return nil, err
	}
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
		return nil, fmt.Errorf("failed to start worker: %w", err)
	}

	return &LLMBatch{
		Opts:   opts,
		cmd:    cmd,
		stdin:  stdin,
		stdout: bufio.NewReader(stdout),
	}, nil
}

func (b *LLMBatch) AddPrompt(prompt string) error {
	// Send the prompt terminated by an <EOL> line to buffer it for batch generation.
	if _, err := io.WriteString(b.stdin, prompt+"\n<EOL>\n"); err != nil {
		return fmt.Errorf("send prompt to worker: %w", err)
	}
	return nil
}

func (b *LLMBatch) Generate() ([]string, error) {
	// Send the batch generation trigger to the worker.
	if _, err := io.WriteString(b.stdin, "<EOF>\n"); err != nil {
		return nil, fmt.Errorf("send batch generation trigger to worker: %w", err)
	}

	var results []string
	var thisResult string
	for {
		line, err := b.stdout.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("read worker output: %w", err)
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "<EOF>" {
			break
		}
		if line == "<EOL>" {
			results = append(results, thisResult)
			thisResult = ""
			continue
		}
		thisResult += line + "\n"
	}
	return results, nil
}

func (b *LLMBatch) Close() error {
	if b.cmd == nil {
		// Already closed or never started: nothing to do.
		return nil
	}
	if _, err := io.WriteString(b.stdin, "<EOF>\n"); err != nil {
		return fmt.Errorf("send EOF to worker: %w", err)
	}
	b.stdin.Close()
	if err := b.cmd.Wait(); err != nil {
		return fmt.Errorf("worker failed: %w", err)
	}
	b.cmd = nil
	return nil
}

// worker creates an exec.Cmd for the worker process with the given chat options and mode.
func (e *LLMEngine) worker(c ChatOpts, mode string) (*exec.Cmd, error) {
	// Prepare the arguments for the worker command.
	args := makeWorkerArgs(e.Opts)
	args = append(args, makeWorkerChatArgs(c)...)
	switch mode {
	case "batch":
		args = append(args, "--batch")
	case "interactive":
		args = append(args, "--interactive")
	}

	// Refuse to run the worker if it is not a regular file (e.g. a planted symlink).
	if info, err := os.Lstat(e.workerPath); err != nil || !info.Mode().IsRegular() {
		return nil, fmt.Errorf("worker executable %q is not a regular file", e.workerPath)
	}

	return exec.Command(e.workerPath, args...), nil
}

/// Helper functions

// Creates a slice with the arguments to pass to the worker
// that control the overall engine behavior. Text
// generation settings have to be specified separately per session.
func makeWorkerArgs(o EngineOpts) []string {
	args := []string{
		"--model", o.ModelPath,
		"--device", o.InferenceDevice.String(),
		"--cpu-pinning", strconv.FormatBool(o.CPUPinning),
		"--num-threads", strconv.Itoa(o.NumThreads),
		"--performance-mode", o.PerformanceMode.String(),
	}
	if o.DebugMode {
		args = append(args, "--debug")
	}
	if isVisionLanguageModel(o.ModelPath) {
		args = append(args, "--vlm")
	}
	return args
}

// Creates a slice with the arguments to pass to the worker
// that control the chat session behavior.
func makeWorkerChatArgs(o ChatOpts) []string {
	args := []string{
		"--max-new-tokens", strconv.FormatUint(uint64(o.MaxTokens), 10),
		"--temperature", fmt.Sprintf("%f", o.Temperature),
		"--max-history", strconv.Itoa(o.MaxHistory),
	}
	if o.SystemPrompt != "" {
		args = append(args, "--system", o.SystemPrompt)
	}
	return args
}

// Try to guess if the given model is a vision-language model based on its contents.
func isVisionLanguageModel(modelPath string) bool {
	// VLM exports carry a vision encoder graph; text-only LLMs do not.
	_, err := os.Stat(filepath.Join(modelPath, "openvino_vision_embeddings_model.xml"))
	return err == nil
}
