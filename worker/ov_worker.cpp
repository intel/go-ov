/**
 * OpenVINO GenAI worker implementation.
 * Hosts the OpenVINO GenAI pipeline and communicates with the Go driver over stdio.
 * 
 * These contents may have been developed with support from one or more Intel-operated generative artificial intelligence solutions.
 * 
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <io.h>
#include <fcntl.h>

#include <openvino/genai/chat_history.hpp>
#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/streamer_base.hpp>
#include <openvino/genai/visual_language/pipeline.hpp>
#include <openvino/runtime/properties.hpp>

static std::string read_all_stdin() {
    _setmode(_fileno(stdin), _O_BINARY);
    std::string data;
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), stdin)) > 0) {
        data.append(buf, n);
    }
    return data;
}

// Argument structure: holds the command-line arguments passed to the worker.
struct Args {
    std::string model_dir;
    std::string device;
    size_t max_new_tokens = 0;
    float temperature = 0.0f;             // optional; 0 = greedy, >0 enables sampling
    int num_threads = 0;                  // optional; 0 lets OpenVINO decide
    bool cpu_pinning = false;             // optional
    std::string perf_mode = "UNDEFINED";  // optional; UNDEFINED leaves the hint unset
    bool debug = false;                   // optional; prints the effective arguments
    bool interactive = false;             // optional; serve prompts in a loop over stdio
    bool batch = false;                   // optional; collect prompts then generate them as one batch
    bool vlm = false;                     // optional; load the model as a vision-language pipeline (text-only)
    std::string system;                   // optional; system prompt applied via the chat template
    int max_history = 10;                 // optional; max retained user/assistant messages
};

// Chat memory: the conversation history passed to the chat template each turn.
struct ChatMemory {
    std::vector<ov::AnyMap> messages;  // messages[0] is the system prompt when present
    size_t system_count = 0;           // 1 if a system prompt anchors the history, else 0
    size_t max_entries = 10;           // max retained user/assistant messages (excludes system)
};


// Parses argv into args; on failure prints a message, sets exit_code, and returns false.
static bool parse_args(int argc, char** argv, Args& args, int& exit_code) {
    std::string max_new_tokens_arg;

    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--debug") {
            args.debug = true;
            continue;
        }
        if (flag == "--interactive") {
            args.interactive = true;
            continue;
        }
        if (flag == "--batch") {
            args.batch = true;
            continue;
        }
        if (flag == "--vlm") {
            args.vlm = true;
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", flag.c_str());
            exit_code = 1;
            return false;
        }
        const char* value = argv[++i];
        if (flag == "--model") {
            args.model_dir = value;
        } else if (flag == "--device") {
            args.device = value;
        } else if (flag == "--max-new-tokens") {
            max_new_tokens_arg = value;
        } else if (flag == "--num-threads") {
            args.num_threads = atoi(value);
        } else if (flag == "--cpu-pinning") {
            args.cpu_pinning = (std::string(value) == "true" || std::string(value) == "1");
        } else if (flag == "--performance-mode") {
            args.perf_mode = value;
        } else if (flag == "--temperature") {
            args.temperature = (float)atof(value);
        } else if (flag == "--system") {
            args.system = value;
        } else if (flag == "--max-history") {
            args.max_history = atoi(value);
        } else {
            fprintf(stderr, "unknown argument: %s\n", flag.c_str());
            exit_code = 1;
            return false;
        }
    }

    if (args.debug) {
        fprintf(stderr, "[debug] model=%s\n", args.model_dir.c_str());
        fprintf(stderr, "[debug] device=%s\n", args.device.c_str());
        fprintf(stderr, "[debug] max-new-tokens=%s\n", max_new_tokens_arg.c_str());
        fprintf(stderr, "[debug] temperature=%f\n", args.temperature);
        fprintf(stderr, "[debug] do-sample=%s\n", args.temperature > 0.0f ? "true (sampling)" : "false (greedy)");
        fprintf(stderr, "[debug] num-threads=%d\n", args.num_threads);
        fprintf(stderr, "[debug] cpu-pinning=%s\n", args.cpu_pinning ? "true" : "false");
        fprintf(stderr, "[debug] performance-mode=%s\n", args.perf_mode.c_str());
        fprintf(stderr, "[debug] interactive=%s\n", args.interactive ? "true" : "false");
        fprintf(stderr, "[debug] batch=%s\n", args.batch ? "true" : "false");
        fprintf(stderr, "[debug] vlm=%s\n", args.vlm ? "true" : "false");
        fprintf(stderr, "[debug] system=%s\n", args.system.c_str());
        fprintf(stderr, "[debug] max-history=%d\n", args.max_history);
    }

    if (args.model_dir.empty() || args.device.empty()) {
        fprintf(stderr, "usage: %s --model <dir> --device <name> [--max-new-tokens <n>] [--temperature <float>] [--num-threads <n>] [--cpu-pinning <true|false>] [--performance-mode <LATENCY|THROUGHPUT>] [--system <text>] [--max-history <n>] [--interactive] [--batch] [--vlm] [--debug]\n", argv[0]);
        exit_code = 1;
        return false;
    }

    args.max_new_tokens = (size_t)strtoull(max_new_tokens_arg.c_str(), nullptr, 10);
    return true;
}

// Builds the OpenVINO plugin config from the parsed arguments.
static ov::AnyMap build_config(const Args& args) {
    ov::AnyMap config;
    config.insert(ov::hint::enable_cpu_pinning(args.cpu_pinning));
    if (args.num_threads > 0) {
        config.insert(ov::inference_num_threads(args.num_threads));
    }
    if (args.perf_mode == "LATENCY") {
        config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    } else if (args.perf_mode == "THROUGHPUT") {
        config.insert(ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    }
    return config;
}

// Builds the generation config from the parsed arguments.
static ov::genai::GenerationConfig build_generation_config(const Args& args) {
    ov::genai::GenerationConfig gen_config;
    if (args.max_new_tokens > 0) {
        gen_config.max_new_tokens = args.max_new_tokens;  // 0/unset leaves the OpenVINO default
    }
    if (args.temperature > 0.0f) {
        gen_config.do_sample = true;
        gen_config.temperature = args.temperature;
    }
    return gen_config;
}

// Builds the chat memory from the parsed arguments, seeding the system prompt when provided.
static ChatMemory build_chat_memory(const Args& args) {
    ChatMemory memory;
    memory.max_entries = args.max_history > 0 ? (size_t)args.max_history : 0;
    if (!args.system.empty()) {
        memory.messages.push_back({{"role", "system"}, {"content", args.system}});
        memory.system_count = 1;
    }
    return memory;
}

// Generator abstracts the backing pipeline (LLM or VLM) behind a uniform chat-history generate
// call so the streaming and output logic below is shared by both.
using Generator = std::function<ov::genai::DecodedResults(
    const ov::genai::ChatHistory&, const ov::genai::GenerationConfig&, ov::genai::StreamerVariant)>;

// Appends the user prompt, streams the generated response to stdout token-by-token as it is
// produced, records the assistant reply (trimmed to the memory limit), and appends <EOL> if requested.
static void generate_and_output(const Generator& gen, const ov::genai::GenerationConfig& gen_config,
                                ChatMemory& memory, const std::string& prompt, bool emit_marker, bool debug) {
    memory.messages.push_back({{"role", "user"}, {"content", prompt}});

    // Write each decoded subword to stdout as soon as it is generated.
    auto streamer = [](std::string subword) -> ov::genai::StreamingStatus {
        fwrite(subword.data(), 1, subword.size(), stdout);
        fflush(stdout);
        return ov::genai::StreamingStatus::RUNNING;
    };

    ov::genai::ChatHistory history(memory.messages);
    ov::genai::DecodedResults decoded = gen(history, gen_config, streamer);
    const std::string result = decoded;

    if (debug) {
        ov::genai::PerfMetrics& pm = decoded.perf_metrics;
        fprintf(stderr, "[metrics] input_tokens=%zu generated_tokens=%zu\n",
                pm.get_num_input_tokens(), pm.get_num_generated_tokens());
        fprintf(stderr, "[metrics] ttft_ms=%.1f tpot_ms=%.2f throughput_tok_s=%.1f generate_ms=%.1f\n",
                pm.get_ttft().mean, pm.get_tpot().mean, pm.get_throughput().mean, pm.get_generate_duration().mean);
    }

    memory.messages.push_back({{"role", "assistant"}, {"content", result}});

    // Keep the system prompt plus the most recent max_entries messages.
    if (memory.messages.size() > memory.system_count + memory.max_entries) {
        const size_t remove = memory.messages.size() - (memory.system_count + memory.max_entries);
        memory.messages.erase(memory.messages.begin() + memory.system_count,
                              memory.messages.begin() + memory.system_count + remove);
    }

    // The streamer already emitted the response; only the terminator remains.
    if (emit_marker) {
        fputs("\n<EOL>\n", stdout);
    }
    fflush(stdout);
}

// Interactive loop: accumulates prompt lines until <EOL>, generates a response for each, and
// reuses the same pipeline until <EOL> is replaced by <EOF> (or stdin closes), which exits.
static void run_interactive(const Generator& gen, const ov::genai::GenerationConfig& gen_config,
                            ChatMemory& memory, bool debug) {
    _setmode(_fileno(stdin), _O_BINARY);

    std::string prompt;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();  // tolerate CRLF line endings
        }
        if (line == "<EOL>" || line == "<EOF>") {
            if (!prompt.empty()) {
                generate_and_output(gen, gen_config, memory, prompt, true, debug);
                prompt.clear();
            }
            if (line == "<EOF>") {
                return;
            }
            continue;
        }
        if (!prompt.empty()) {
            prompt += "\n";
        }
        prompt += line;
    }
    // stdin closed with a pending prompt: treat it as an implicit <EOL>.
    if (!prompt.empty()) {
        generate_and_output(gen, gen_config, memory, prompt, true, debug);
    }
}

// Batch loop: collects prompts (each terminated by <EOL>) until <EOF> or stdin close, then
// generates the whole set in a single batched call. Each prompt is rendered through the chat
// template (with the optional system prompt) so results match interactive mode. Responses are
// written back in input order, each terminated by <EOL>, with a final <EOF> closing the batch.
static void run_batch(ov::genai::LLMPipeline& pipe, const ov::genai::GenerationConfig& gen_config,
                      const Args& args, bool debug) {
    _setmode(_fileno(stdin), _O_BINARY);

    std::vector<std::string> prompts;
    std::string prompt;
    std::string line;
    bool done = false;
    while (!done && std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();  // tolerate CRLF line endings
        }
        if (line == "<EOL>" || line == "<EOF>") {
            if (!prompt.empty()) {
                prompts.push_back(prompt);  // a partial prompt at <EOF> still counts
                prompt.clear();
            }
            if (line == "<EOF>") {
                done = true;  // <EOF> ends the batch and triggers processing
            }
            continue;
        }
        if (!prompt.empty()) {
            prompt += "\n";
        }
        prompt += line;
    }
    // stdin closed with a pending prompt: treat it as a final prompt.
    if (!prompt.empty()) {
        prompts.push_back(prompt);
    }

    if (prompts.empty()) {
        fputs("<EOF>\n", stdout);
        fflush(stdout);
        return;
    }

    // Render each prompt with the chat template up front so the batched string inputs carry the
    // system prompt; disable the pipeline's own templating to avoid double-wrapping.
    ov::genai::Tokenizer tokenizer = pipe.get_tokenizer();
    std::vector<std::string> inputs;
    inputs.reserve(prompts.size());
    for (const std::string& p : prompts) {
        std::vector<ov::AnyMap> messages;
        if (!args.system.empty()) {
            messages.push_back({{"role", "system"}, {"content", args.system}});
        }
        messages.push_back({{"role", "user"}, {"content", p}});
        ov::genai::ChatHistory history(messages);
        inputs.push_back(tokenizer.apply_chat_template(history, true));
    }

    ov::genai::GenerationConfig batch_config = gen_config;
    batch_config.apply_chat_template = false;  // inputs are already templated

    ov::genai::DecodedResults decoded = pipe.generate(inputs, batch_config);

    if (debug) {
        ov::genai::PerfMetrics& pm = decoded.perf_metrics;
        fprintf(stderr, "[metrics] batch_size=%zu input_tokens=%zu generated_tokens=%zu\n",
                prompts.size(), pm.get_num_input_tokens(), pm.get_num_generated_tokens());
        fprintf(stderr, "[metrics] ttft_ms=%.1f tpot_ms=%.2f throughput_tok_s=%.1f generate_ms=%.1f\n",
                pm.get_ttft().mean, pm.get_tpot().mean, pm.get_throughput().mean, pm.get_generate_duration().mean);
    }

    // results.texts[i] corresponds to inputs[i].
    for (const std::string& text : decoded.texts) {
        fwrite(text.data(), 1, text.size(), stdout);
        fputs("\n<EOL>\n", stdout);
    }
    fputs("<EOF>\n", stdout);
    fflush(stdout);
}

int main(int argc, char** argv) {
    Args args;
    int exit_code = 0;
    if (!parse_args(argc, argv, args, exit_code)) {
        return exit_code;
    }

    try {
        ov::AnyMap config = build_config(args);

        ov::genai::GenerationConfig gen_config = build_generation_config(args);

        ChatMemory memory = build_chat_memory(args);

        // Build the generate function backed by the requested pipeline. The LLM pipeline is kept
        // for batch mode, which the VLM pipeline does not support.
        std::shared_ptr<ov::genai::LLMPipeline> llm;
        std::shared_ptr<ov::genai::VLMPipeline> vlm;
        Generator gen;
        if (args.vlm) {
            vlm = std::make_shared<ov::genai::VLMPipeline>(args.model_dir, args.device, config);
            gen = [vlm](const ov::genai::ChatHistory& history, const ov::genai::GenerationConfig& cfg,
                        ov::genai::StreamerVariant streamer) -> ov::genai::DecodedResults {
                return vlm->generate(history, std::vector<ov::Tensor>{}, cfg, streamer);
            };
        } else {
            llm = std::make_shared<ov::genai::LLMPipeline>(args.model_dir, args.device, config);
            gen = [llm](const ov::genai::ChatHistory& history, const ov::genai::GenerationConfig& cfg,
                        ov::genai::StreamerVariant streamer) -> ov::genai::DecodedResults {
                return llm->generate(history, cfg, streamer);
            };
        }

        _setmode(_fileno(stdout), _O_BINARY);

        if (args.batch) {
            if (!llm) {
                fprintf(stderr, "batch mode is not supported for VLM models\n");
                return 1;
            }
            run_batch(*llm, gen_config, args, args.debug);
        } else if (args.interactive) {
            run_interactive(gen, gen_config, memory, args.debug);
        } else {
            const std::string prompt = read_all_stdin();
            generate_and_output(gen, gen_config, memory, prompt, false, args.debug);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 2;
    }

    return 0;
}
