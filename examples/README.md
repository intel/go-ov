# go-ov Examples

This folder contains small, self-contained programs demonstrating the main ways
to use the `go-ov` library. Each example is its own `package main` in its own
directory.

## Available Examples

| Example | Directory | Description |
|---|---|---|
| One-shot generation | [generate](generate) | Generate a single response from a prompt, no persistent session. |
| One-shot generation (streaming) | [generate-streaming](generate-streaming) | Same as above, printing tokens as they are produced via `engine.Stream`. |
| Session with memory | [session](session) | Persistent multi-turn session that remembers earlier turns. |
| Session with memory (streaming) | [session-streaming](session-streaming) | Same as above, streaming each turn's response via `session.Stream`. |
| Batch generation | [batch](batch) | Load several prompts, then generate all responses in one pass. |

## Prerequisites

All examples expect, relative to the directory you run them from:

- `models/OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov` — an OpenVINO IR model. See
  [util/download-model.sh](../util/download-model.sh).
- `bin` — a directory with the OpenVINO GenAI runtime DLLs. See
  [util/make-ov-runtime.sh](../util/make-ov-runtime.sh).

## Running

Run from the repository root so the relative `models/` and `bin` paths resolve:

```bash
go run ./examples/generate
go run ./examples/generate-streaming
go run ./examples/session
go run ./examples/session-streaming
go run ./examples/batch
```
