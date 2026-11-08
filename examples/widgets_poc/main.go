// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT

package main

import (
	"fmt"
	"path/filepath"
	"strings"
	"text/template"
	"time"

	goov "github.com/intel/go-ov"
)

const SYSTEM_PROMPT = `
You help users of a laptop battery-life diagnostic tool (Intel-based Windows systems)
understand the results of a single analysis run. The report is divided into sections
called widgets.

Each turn you receive a widget's title, its purpose, the metrics measured in THIS run,
any findings (notable or abnormal behavior detected), and definitions of relevant terms.

Your job is to interpret this run's results in plain, natural language:
- Lead with what the measured values and findings mean for this system — is the result
  good, concerning, or neutral, and why.
- Always ground statements in the specific values from this run.
- When findings are present, explain them and their practical impact on power, battery
  life, or performance, using only the information provided.

Do NOT restate the widget's purpose, re-explain what a metric is, or repeat the
definitions — the user already sees those. Do not add facts that aren't in the provided
details. Be concise and specific; skip generic background and filler.
`
const AI_MAX_TOKENS = 500   // 250 on original qwen2
const AI_TEMPERATURE = 0    // ensures the model generates deterministic responses
const AI_MAX_HISTORY = 0    // each widget evaluation is independent from each other
const AI_DEBUG_MODE = false // set to true to enable debug mode

const WIDGET_PROMPT = `
Interpret the results of this widget for the current analysis run.

Widget: {{.Title}}
Purpose (background — understand, do not repeat): {{.Description}}
Term definitions (background — understand, do not repeat): {{.Definitions}}

Results to interpret:
- Measured metrics: {{.Metrics}}
- Findings: {{.Hints}}

In 2–4 natural sentences, explain what these specific results mean for this system,
emphasizing the findings and the measured values. If there are no findings, briefly say
whether the values look nominal based only on what's provided. Do not restate the purpose
or define terms.
`

// Minimal prompt whose only job is to trigger the first (cold) inference so real turns run warm.
const WARMUP_PROMPT = `Write two short sentences confirming the engine is ready. /no_think`

type widgetInfo struct {
	Title       string
	Description string
	Metrics     string
	Definitions string // should include definitions for each relevant term
	Hints       string
}

func main() {
	start := time.Now()
	cwd, err := filepath.Abs(".")
	if err != nil {
		panic(err)
	}

	e, err := initEngine(cwd)
	if err != nil {
		panic(err)
	}

	ibh := widgetInfo{
		Title:       "Idle/Busy Histogram",
		Description: "Displays an aggregated analysis of overall CPU Idle and Busy periods during the assessment period. Broadly, CPU 'idle-ness' buckets (or periods) refer to the interval between periods of CPU activity or quiet periods. Conversely, CPU 'busy-ness' buckets refers to periods in which the CPU was actively being used by a process. For example, an 'Idle' bucket of '<=5ms' refer to the sum of all CPU 'quiet' times between 1ms and 5ms. A 'busy' bucket of '<=5ms' refer to the sum of all the periods in which the CPU was being actively used for more than 1ms but less than 5ms. In general, when observing a system on its 'idle' state, meaning that the system is not being actively used, higher 'idle' bucket total times (>100ms) and lower 'busy' bucket total times (<16ms) are desirable and would correlate to lower energy utilization and longer battery life. In other words, this would mean that the CPU spent most time being inactive and is only active in short bursts.",
		Metrics: `
			Total CPU Idle Time (%): 43.8047
		`,
		Hints: `
			CPU Idle time is less than 55%, this may indicate high activity on background processes. Review activity of applications and processes.
		`,
		Definitions: `
			Total CPU Idle Time (%): Percentage of time the CPU spent idle during the assessment period
		`,
	}

	ccu := widgetInfo{
		Title:       "CPU Core Utilization",
		Description: "Information about the utilization of the CPU by each type of core (Performance, Efficiency or Low-Power Efficiency). Processes that consume more CPU usage will likely correlate to higher power usage. Less CPU usage of each process contributes to higher idle duration and lower power. Generally, for background applications and resident proceses, like VPN, Telemetry, Security or other management software, a lower CPU usage (under 0.05%) is desirable. However, this may be different depending on the specific software and usage scenario. The table will show a row for each one of the available core types present on the system. Each row will provide information about the aggregated busy duration (total and average) and the number of cores for each type. The right side displays a pie chart to compare the CPU utilization for each core type.",
		Metrics: `
			Avg Busy Duration of Performance Cores: 3.076, 
			Avg Busy Duration of Efficiency Cores: 39.177, 
			Avg Busy Duration of Low-Power Efficiency Cores: 57.746,
		`,
		Hints: "None",
		Definitions: `
			Avg Busy Duration of Performance Cores: Average percentage of time the Performance cores were busy during the assessment period.
			Avg Busy Duration of Efficiency Cores: Average percentage of time the Efficiency cores were busy during the assessment period.
			Avg Busy Duration of Low-Power Efficiency Cores: Average percentage of time the Low-Power Efficiency cores were busy during the assessment period.
			Performance Cores: Cores designed for high performance tasks, typically consuming more power.
			Efficiency Cores: Cores designed for energy-efficient processing, suitable for background tasks.
			Low-Power Efficiency Cores: Cores designed for low-power, energy-efficient processing, often used for background and low-priority tasks.
		`,
	}

	report := []widgetInfo{
		ibh,
		ccu,
	}

	method := "session_stream"
	switch method {
	case "standalone":
		evaluateWithEngine(report, e)
	case "standalone_stream":
		evaluateWithEngineStream(report, e)
	case "session":
		evaluateWithSession(report, e)
	case "session_stream":
		evaluateWithSessionStream(report, e)
	case "batch":
		evaluateWithBatch(report, e)
	}

	fmt.Printf("\n-----------------------------------\n")
	fmt.Printf("Total time taken: %s\n", time.Since(start))
	fmt.Printf("-----------------------------------\n")
}

func evaluateWithEngine(widgets []widgetInfo, e *goov.LLMEngine) {
	for _, w := range widgets {
		evaluateWidgetWithEngine(w.Title, e, w, false)
	}
}

func evaluateWithEngineStream(widgets []widgetInfo, e *goov.LLMEngine) {
	for _, w := range widgets {
		evaluateWidgetWithEngine(w.Title, e, w, true)
	}
}

func evaluateWithSession(widgets []widgetInfo, e *goov.LLMEngine) {
	s, _ := initSession("S_BLOCK", e)
	defer s.Close()
	for _, w := range widgets {
		evaluateWidgetWithSession(w.Title, s, w, false)
	}
}

func evaluateWithSessionStream(widgets []widgetInfo, e *goov.LLMEngine) {
	s, _ := initSession("S_STREAM", e)
	defer s.Close()
	for _, w := range widgets {
		evaluateWidgetWithSession(w.Title, s, w, true)
	}
}

func evaluateWithBatch(widgets []widgetInfo, e *goov.LLMEngine) {
	b, _ := initBatch("BATCH", e)
	defer b.Close()
	for _, w := range widgets {
		prompt, _ := w.buildPrompt()
		b.AddPrompt(prompt)
	}
	bResponses, err := b.Generate()
	if err != nil {
		panic(err)
	}
	for i, resp := range bResponses {
		fmt.Printf("Batch Response %d: %s\n", i+1, resp)
	}
}

func evaluateWidgetWithEngine(name string, e *goov.LLMEngine, w widgetInfo, stream bool) {
	prompt, err := w.buildPrompt()
	if err != nil {
		panic(err)
	}

	opts := goov.ChatOpts{
		MaxTokens:    AI_MAX_TOKENS,
		Temperature:  AI_TEMPERATURE,
		SystemPrompt: SYSTEM_PROMPT,
		MaxHistory:   AI_MAX_HISTORY,
	}

	start := time.Now()
	fmt.Printf("\n> %s (%s)\n", name, w.Title)
	if stream {
		streamCh, err := e.Stream(prompt, opts)
		if err != nil {
			panic(err)
		}
		for chunk := range streamCh {
			if chunk.Err != nil {
				panic(chunk.Err)
			}
			fmt.Print(chunk.Text)
		}
		fmt.Println()
	} else {
		response, err := e.Generate(prompt, opts)
		if err != nil {
			panic(err)
		}
		println(response)
	}
	fmt.Printf("Time taken for %s: %s\n", name, time.Since(start))
}

func evaluateWidgetWithSession(name string, s *goov.LLMSession, w widgetInfo, stream bool) {
	prompt, err := w.buildPrompt()
	if err != nil {
		panic(err)
	}

	start := time.Now()
	fmt.Printf("\n> %s (%s)\n", name, w.Title)
	if stream {
		streamCh, err := s.Stream(prompt)
		if err != nil {
			panic(err)
		}
		for chunk := range streamCh {
			if chunk.Err != nil {
				panic(chunk.Err)
			}
			fmt.Print(chunk.Text)
		}
		fmt.Println()
	} else {
		response, err := s.Generate(prompt)
		if err != nil {
			panic(err)
		}
		println(response)
	}
	fmt.Printf("Time taken for %s: %s\n", name, time.Since(start))
}

func (w widgetInfo) buildPrompt() (string, error) {
	tmpl, err := template.New("widget").Parse(WIDGET_PROMPT)
	if err != nil {
		return "", err
	}

	var sb strings.Builder
	if err := tmpl.Execute(&sb, w); err != nil {
		return "", err
	}
	return sb.String(), nil
}

func initEngine(path string) (*goov.LLMEngine, error) {
	opts := goov.DefaultEngineOpts() // or SafeEngineOpts() for doing a low impact CPU-only inference
	// opts.ModelPath = filepath.Join(path, "../go-ov/models/qwen2.5-1.5b-int4")
	// opts.ModelPath = filepath.Join(path, "../go-ov/models/qwen3.5-2b-int4")
	opts.ModelPath = filepath.Join(path, "../go-ov/models/qwen3-1.7b-int4")
	opts.LibraryPath = filepath.Join(path, "../go-ov/bin")
	opts.PerformanceMode = goov.OV_PERFORMANCE_MODE_LATENCY
	opts.InferenceDevice = goov.OV_INFERENCE_DEVICE_GPU
	opts.DebugMode = AI_DEBUG_MODE

	engine, err := goov.New(opts)
	if err != nil {
		return nil, err
	}

	return engine, nil
}

func initSession(name string, engine *goov.LLMEngine) (*goov.LLMSession, error) {
	start := time.Now()
	s, err := engine.NewSession(goov.ChatOpts{
		MaxTokens:    AI_MAX_TOKENS,
		Temperature:  AI_TEMPERATURE,
		SystemPrompt: SYSTEM_PROMPT,
		MaxHistory:   AI_MAX_HISTORY,
	})
	if err != nil {
		return nil, err
	}

	// quick throwaway generation to warm up the engine
	if _, err = s.Generate(WARMUP_PROMPT); err != nil {
		return nil, err
	}

	fmt.Printf("Session %s initialized in %s\n", name, time.Since(start))

	return s, nil
}

func initBatch(name string, engine *goov.LLMEngine) (*goov.LLMBatch, error) {
	start := time.Now()
	b, err := engine.NewBatch(goov.ChatOpts{
		MaxTokens:    AI_MAX_TOKENS,
		Temperature:  AI_TEMPERATURE,
		SystemPrompt: SYSTEM_PROMPT,
	})
	if err != nil {
		return nil, err
	}

	fmt.Printf("Batch %s initialized in %s\n", name, time.Since(start))

	return b, nil
}
