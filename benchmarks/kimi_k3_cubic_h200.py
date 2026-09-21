# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay chat workloads with semantic TTFT and token-weighted cache metrics.

This client needs only the standard library. The optional prepare command needs
the tokenizer dependencies already installed by vLLM. Run on a dedicated server:
each round resets its prefix cache, after compilation/graph warmup has completed.
"""

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import statistics
import time
import urllib.request
from pathlib import Path

MODEL_REVISION = "f29f15dc4afd99feb3349b538bbe7ed439787853"
HISTOGRAMS = ("inter_token_latency_seconds", "request_queue_time_seconds")
COUNTERS = (
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:num_preemptions_total",
) + tuple(f"vllm:{name}_{suffix}" for name in HISTOGRAMS for suffix in ("sum", "count"))


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def consume_stream(lines, started, clock=time.perf_counter):
    """Ignore role/empty/usage frames; require semantic output and full closure."""
    first = first_content = last = None
    finish = usage = None
    done = False
    content, reasoning, gaps = [], [], []
    for raw in lines:
        line = raw.decode("utf-8").strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
            break
        event = json.loads(payload)
        if event.get("error"):
            raise ValueError(f"SSE error: {event['error']}")
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            answer = delta.get("content") or ""
            thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if answer or thought:
                now = clock()
                if first is None:
                    first = now
                if answer and first_content is None:
                    first_content = now
                if last is not None:
                    gaps.append((now - last) * 1000)
                last = now
                content.append(answer)
                reasoning.append(thought)
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
    output_tokens = (usage or {}).get("completion_tokens", 0)
    success = (
        first is not None
        and done
        and finish in ("stop", "length")
        and output_tokens > 0
    )
    return {
        "success": success,
        "error": None
        if success
        else "Incomplete stream, missing usage, or empty output",
        "ttft_ms": None if first is None else (first - started) * 1000,
        "first_content_ms": (
            None if first_content is None else (first_content - started) * 1000
        ),
        "tpot_ms": (
            (last - first) * 1000 / (output_tokens - 1)
            if first is not None and output_tokens > 1
            else None
        ),
        # Speculative SSE frames may contain several tokens: these are not ITLs.
        "inter_chunk_ms": gaps,
        "latency_ms": (clock() - started) * 1000,
        "usage": usage,
        "finish_reason": finish,
        "done": done,
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
    }


def headers():
    result = {"Content-Type": "application/json"}
    if key := os.environ.get("OPENAI_API_KEY"):
        result["Authorization"] = f"Bearer {key}"
    return result


def request(base, path, payload=None, timeout=1800):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers())
    return urllib.request.urlopen(req, timeout=timeout)


def infer(base, model, body, timeout):
    payload = {
        **body,
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    try:
        with request(base, "/v1/chat/completions", payload, timeout) as response:
            result = consume_stream(response, start)
            result["request_id"] = response.headers.get("x-request-id")
            return result
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "latency_ms": (time.perf_counter() - start) * 1000,
        }


def batch(args, bodies):
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        results = list(
            pool.map(
                lambda body: infer(args.base_url, args.model, body, args.timeout),
                bodies,
            )
        )
    return results, time.perf_counter() - started


def metrics(base):
    with request(base, "/metrics", timeout=30) as response:
        return parse_metrics(response.read().decode())


def reset_cache(base):
    with request(base, "/reset_prefix_cache", {}) as response:
        raw = response.read().strip()
    result = json.loads(raw) if raw else True  # Legacy successful empty response.
    success = result.get("success") if isinstance(result, dict) else result
    if success is not True:
        raise RuntimeError("Server refused prefix-cache reset")


def server_cache_config(base):
    # Whitelist serving properties; never persist server environment/auth data.
    with request(base, "/server_info?config_format=json", timeout=30) as response:
        config = json.load(response)["vllm_config"]
    cache = config["cache_config"]
    result = {
        key: cache.get(key)
        for key in (
            "cache_dtype",
            "block_size",
            "mamba_block_size",
            "num_gpu_blocks",
            "kv_cache_memory_bytes",
            "prefix_match_unit",
            "mamba_cache_mode",
            "prefix_cache_retention_interval",
        )
    }
    result["effective_max_model_len"] = config["model_config"]["max_model_len"]
    return result


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split()[0]
        if name in COUNTERS:
            values[name] = values.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return values


def metric_delta(before, after):
    result = {}
    for name in COUNTERS:
        if name in before and name in after:
            value = after[name] - before[name]
            if value < 0:
                raise ValueError("Metrics reset during measurement; discard this round")
            result[name] = value
    for scope in ("prefix", "external_prefix"):
        hits = result.get(f"vllm:{scope}_cache_hits_total")
        queries = result.get(f"vllm:{scope}_cache_queries_total")
        result[f"{scope}_hit_rate"] = (
            hits / queries if hits is not None and queries else None
        )
    for name in HISTOGRAMS:
        total = result.get(f"vllm:{name}_sum")
        count = result.get(f"vllm:{name}_count")
        result[f"{name}_mean_ms"] = (
            1000 * total / count if total is not None and count else None
        )
    return result


def summarize(results, elapsed):
    good = [r for r in results if r["success"]]
    summary = {
        "requests": len(results),
        "successful": len(good),
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": sum(r["usage"]["completion_tokens"] for r in good)
        / elapsed,
    }
    for field in ("ttft_ms", "first_content_ms", "tpot_ms"):
        values = [r[field] for r in good if r.get(field) is not None]
        summary[field] = {f"p{p}": percentile(values, p) for p in (50, 95, 99)}
    return summary


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.revision, trust_remote_code=True
    )
    paragraph = (
        "Record: a service keeps request prefixes in memory. "
        "Measure queueing, prefill, decode and cache reuse independently. "
    )
    tokens = tokenizer.encode(paragraph, add_special_tokens=False)
    doc = tokenizer.decode(
        (tokens * (args.input_tokens // len(tokens) + 1))[: args.input_tokens - 128]
    )
    rows = []
    for i in range(args.count):
        body = {
            "messages": [
                {"role": "system", "content": "Read the records and answer precisely."},
                {
                    "role": "user",
                    "content": (
                        doc
                        + f"\nRequest {i}: explain prefix reuse and its limitations."
                    ),
                },
            ],
            "temperature": 0,
            "seed": 42,
            "max_tokens": args.output_tokens,
            "ignore_eos": True,
        }
        rows.append(
            {
                "body": body,
                "prompt_tokens": len(
                    tokenizer.apply_chat_template(
                        body["messages"], add_generation_prompt=True
                    )
                ),
            }
        )
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))


def run(args):
    fixtures = [
        json.loads(line)
        for line in args.requests.read_text().splitlines()
        if line.strip()
    ]
    if not fixtures:
        raise ValueError("Request fixture is empty")
    manifest = json.loads(args.server_manifest.read_text())
    report = {
        "schema_version": 1,
        "label": args.label,
        "server": manifest,
        "resolved_cache_config": server_cache_config(args.base_url),
        "fixture_sha256": digest(fixtures),
        "scenario": args.scenario,
        "concurrency": args.concurrency,
        "rounds": [],
    }
    # Compile/capture warmup is outside the prefix-cache experiment.
    warmup, _ = batch(args, [fixtures[0]["body"]])
    if not all(r["success"] for r in warmup):
        raise RuntimeError("Initial warmup failed")
    for round_index in range(args.rounds):
        reset_cache(args.base_url)
        bodies = [copy.deepcopy(row["body"]) for row in fixtures]
        if args.scenario == "cold":
            for i, body in enumerate(bodies):
                body["messages"][0]["content"] = (
                    f"Independent request {round_index}:{i}. "
                    + body["messages"][0]["content"]
                )
        else:
            seeds = bodies[:1] if args.scenario == "shared" else bodies
            warm, _ = batch(args, seeds)
            if not all(r["success"] for r in warm):
                raise RuntimeError("Cache warmup failed")
            if args.scenario == "append":
                for body, previous in zip(bodies, warm, strict=True):
                    body["messages"].extend(
                        [
                            {
                                "role": "assistant",
                                "content": previous["content"],
                                "reasoning_content": previous["reasoning_content"],
                            },
                            {"role": "user", "content": "Give one concrete example."},
                        ]
                    )
            if args.scenario == "pressure":
                distractors = []
                for i in range(args.pressure_requests):
                    body = copy.deepcopy(bodies[i % len(bodies)])
                    body["messages"][0]["content"] = f"Unrelated record {i}."
                    body["max_tokens"] = 1
                    distractors.append(body)
                pressure, _ = batch(args, distractors)
                if not all(r["success"] for r in pressure):
                    raise RuntimeError("Cache-pressure phase failed")
        # Engine statistics are published periodically, independently of SSE.
        time.sleep(args.metrics_settle_seconds)
        before = metrics(args.base_url)
        results, elapsed = batch(args, bodies)
        time.sleep(args.metrics_settle_seconds)
        after = metrics(args.base_url)
        report["rounds"].append(
            {
                "workload_sha256": digest(bodies),
                "summary": summarize(results, elapsed),
                "cache": metric_delta(before, after),
                "results": results,
            }
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(not r["success"] for trial in report["rounds"] for r in trial["results"]):
        raise RuntimeError(f"Failed requests recorded in {args.output}")


def compare(baseline, candidate):
    for key in ("fixture_sha256", "scenario", "concurrency"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Cannot compare different {key}")
    for key in ("model_revision", "tokenizer_revision", "gpu_devices"):
        if not baseline["server"].get(key) or (
            baseline["server"][key] != candidate["server"].get(key)
        ):
            raise ValueError(f"Missing or different server {key}")
    a, b = baseline["rounds"], candidate["rounds"]
    if len(a) != len(b) or len(a) < 3:
        raise ValueError(
            "Comparison requires the same number of rounds, at least three"
        )
    for old, new in zip(a, b, strict=True):
        if old["workload_sha256"] != new["workload_sha256"]:
            raise ValueError(
                "Measured prompts differ (including generated chat history)"
            )
        for trial in (old, new):
            summary = trial["summary"]
            if not summary["requests"] or summary["successful"] != summary["requests"]:
                raise ValueError("Failed requests invalidate performance comparison")
            measurements = (
                summary["ttft_ms"]["p95"],
                summary["output_tokens_per_second"],
            )
            if any(v is None or not math.isfinite(v) or v <= 0 for v in measurements):
                raise ValueError("Comparison requires finite positive measurements")
    ttft_a = statistics.median(r["summary"]["ttft_ms"]["p95"] for r in a)
    ttft_b = statistics.median(r["summary"]["ttft_ms"]["p95"] for r in b)
    throughput_a = statistics.median(
        r["summary"]["output_tokens_per_second"] for r in a
    )
    throughput_b = statistics.median(
        r["summary"]["output_tokens_per_second"] for r in b
    )
    return {
        "p95_ttft_ratio": ttft_b / ttft_a,
        "output_throughput_ratio": throughput_b / throughput_a,
        "performance_gate_passed": ttft_b < ttft_a
        and throughput_b >= 0.9 * throughput_a,
        "requires_separate_gpu_correctness_and_quality_pass": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--tokenizer", required=True)
    prep.add_argument("--revision", default=MODEL_REVISION)
    prep.add_argument("--input-tokens", type=int, default=8192)
    prep.add_argument("--output-tokens", type=int, default=1024)
    prep.add_argument("--count", type=int, default=128)
    prep.add_argument("--output", type=Path, required=True)
    bench = commands.add_parser("run")
    bench.add_argument("--base-url", default="http://127.0.0.1:8000")
    bench.add_argument("--model", default="Kimi-K3-Cubic-2.5Bit")
    bench.add_argument("--requests", type=Path, required=True)
    bench.add_argument("--server-manifest", type=Path, required=True)
    bench.add_argument("--label", required=True)
    bench.add_argument(
        "--scenario",
        choices=("cold", "repeat", "shared", "append", "pressure"),
        default="cold",
    )
    bench.add_argument("--concurrency", type=int, default=8)
    bench.add_argument("--rounds", type=int, default=3)
    bench.add_argument("--pressure-requests", type=int, default=256)
    bench.add_argument("--timeout", type=float, default=1800)
    bench.add_argument("--metrics-settle-seconds", type=float, default=6)
    bench.add_argument("--output", type=Path, required=True)
    diff = commands.add_parser("compare")
    diff.add_argument("baseline", type=Path)
    diff.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.input_tokens < 256 or args.count < 1 or args.output_tokens < 1:
            parser.error(
                "Input must be >=256 tokens and request/output counts positive"
            )
        prepare(args)
    elif args.command == "run":
        if args.concurrency < 1 or args.rounds < 1 or args.pressure_requests < 1:
            parser.error("Concurrency, rounds and pressure requests must be positive")
        if args.timeout <= 0 or args.metrics_settle_seconds < 0:
            parser.error("Timeout must be positive and metrics settling non-negative")
        run(args)
    else:
        result = compare(
            json.loads(args.baseline.read_text()),
            json.loads(args.candidate.read_text()),
        )
        print(json.dumps(result, indent=2))
        if not result["performance_gate_passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
