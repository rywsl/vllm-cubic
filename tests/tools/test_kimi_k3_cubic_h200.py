# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.kimi_k3_cubic_h200 import (
    compare,
    consume_stream,
    metric_delta,
    parse_metrics,
    reset_cache,
    run,
)


def event(delta=None, finish=None, usage=None):
    payload: dict[str, Any] = {"choices": []}
    if delta is not None or finish is not None:
        payload["choices"] = [{"delta": delta or {}, "finish_reason": finish}]
    if usage is not None:
        payload["usage"] = usage
    return "data: " + json.dumps(payload)


def complete_stream():
    return [
        ": keepalive",
        event({"role": "assistant", "content": ""}),
        event({"reasoning_content": "Check the record."}),
        event({"content": "Answer"}),
        event(finish="length"),
        event(usage={"prompt_tokens": 4096, "completion_tokens": 3}),
        "data: [DONE]",
    ]


def test_ttft_ignores_role_and_includes_reasoning_before_visible_answer():
    clock = iter([11.0, 12.0, 13.0])
    result = consume_stream(complete_stream(), 10.0, lambda: next(clock))
    assert result["success"]
    assert result["ttft_ms"] == 1000
    assert result["first_content_ms"] == 2000
    assert result["tpot_ms"] == 500
    assert result["inter_chunk_ms"] == [1000]


@pytest.mark.parametrize("omit", [2, 5, 6])
def test_incomplete_or_empty_stream_never_counts_as_success(omit):
    lines = complete_stream()
    if omit == 2:
        lines = [line for i, line in enumerate(lines) if i not in (2, 3)]
    else:
        lines.pop(omit)
    assert not consume_stream(lines, 0, lambda: 1)["success"]


def test_usage_on_finish_frame_is_counted_and_sse_errors_fail():
    result = consume_stream(
        [
            event({"content": "a"}),
            event(finish="stop", usage={"completion_tokens": 1}),
            "data: [DONE]",
        ],
        0,
        lambda: 1,
    )
    assert result["success"]
    with pytest.raises(ValueError, match="SSE error"):
        consume_stream(['data: {"error":{"message":"overloaded"}}'], 0)


def test_cache_ratios_are_token_weighted_across_ranks_and_separate_external():
    before = parse_metrics(
        'vllm:prefix_cache_hits_total{engine="0"} 10\n'
        'vllm:prefix_cache_queries_total{engine="0"} 20\n'
    )
    after = parse_metrics(
        'vllm:prefix_cache_hits_total{engine="0"} 30\n'
        'vllm:prefix_cache_hits_total{engine="1"} 40\n'
        'vllm:prefix_cache_queries_total{engine="0"} 60\n'
        'vllm:prefix_cache_queries_total{engine="1"} 60\n'
    )
    delta = metric_delta(before, after)
    assert delta["prefix_hit_rate"] == 0.6
    assert delta["external_prefix_hit_rate"] is None
    with pytest.raises(ValueError, match="Metrics reset"):
        metric_delta(after, before)


@pytest.mark.parametrize("payload", [b'{"success": false}', b"false", b"{}"])
def test_failed_cache_reset_is_not_mislabeled_as_cold(monkeypatch, payload):
    monkeypatch.setattr(
        "benchmarks.kimi_k3_cubic_h200.request",
        lambda *args: io.BytesIO(payload),
    )
    with pytest.raises(RuntimeError, match="refused"):
        reset_cache("http://127.0.0.1:8000")


@pytest.mark.parametrize("payload", [b'{"success": true}', b"true", b""])
def test_cache_reset_accepts_current_and_legacy_success(monkeypatch, payload):
    monkeypatch.setattr(
        "benchmarks.kimi_k3_cubic_h200.request",
        lambda *args: io.BytesIO(payload),
    )
    reset_cache("http://127.0.0.1:8000")


def report(ttft=100, throughput=100):
    return {
        "server": {
            "model_revision": "target",
            "tokenizer_revision": "tokenizer",
            "gpu_devices": [{"name": "H200"}],
        },
        "fixture_sha256": "fixture",
        "scenario": "cold",
        "concurrency": 8,
        "rounds": [
            {
                "workload_sha256": "actual-prompts",
                "summary": {
                    "successful": 128,
                    "requests": 128,
                    "ttft_ms": {"p95": ttft},
                    "output_tokens_per_second": throughput,
                },
            }
            for _ in range(3)
        ],
    }


@pytest.mark.parametrize("throughput, passed", [(90, True), (89, False)])
def test_latency_priority_still_enforces_ten_percent_throughput_floor(
    throughput, passed
):
    result = compare(report(), report(ttft=80, throughput=throughput))
    assert result["performance_gate_passed"] is passed
    assert result["requires_separate_gpu_correctness_and_quality_pass"]


@pytest.mark.parametrize(
    "change", ["hardware", "history", "failure", "rounds", "zero", "nan"]
)
def test_invalid_comparisons_are_rejected(change):
    baseline = report()
    candidate = copy.deepcopy(baseline)
    if change == "hardware":
        candidate["server"]["gpu_devices"] = [{"name": "B300"}]
    elif change == "history":
        candidate["rounds"][0]["workload_sha256"] = "different-answer-history"
    elif change == "failure":
        candidate["rounds"][0]["summary"]["successful"] = 127
    elif change in ("zero", "nan"):
        candidate["rounds"][0]["summary"]["ttft_ms"]["p95"] = (
            0 if change == "zero" else float("nan")
        )
    else:
        candidate["rounds"].pop()
    with pytest.raises(ValueError):
        compare(baseline, candidate)


def test_chat_replay_records_complete_rounds_and_whitelists_server_info(tmp_path):
    state = {"requests": 0, "resets": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, data, content_type="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/server_info"):
                self.respond(
                    json.dumps(
                        {
                            "vllm_config": {
                                "cache_config": {"cache_dtype": "fp8_q16"},
                                "model_config": {"max_model_len": 131072},
                            },
                            "vllm_env": {"irrelevant_secret": "must-not-persist"},
                        }
                    ).encode()
                )
            else:
                self.respond(
                    (
                        f"vllm:prefix_cache_queries_total {state['requests'] * 4096}\n"
                        "vllm:prefix_cache_hits_total 0\n"
                    ).encode(),
                    "text/plain",
                )

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/reset_prefix_cache":
                state["resets"] += 1
                self.respond(b'{"success":true}')
            else:
                assert payload["stream_options"]["include_usage"]
                state["requests"] += 1
                self.respond(
                    "\n\n".join(complete_stream()).encode(), "text/event-stream"
                )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fixtures = tmp_path / "requests.jsonl"
    fixtures.write_text(
        json.dumps(
            {
                "body": {
                    "messages": [
                        {"role": "system", "content": "a"},
                        {"role": "user", "content": "b"},
                    ],
                    "max_tokens": 3,
                }
            }
        )
        + "\n"
    )
    manifest = tmp_path / "server.json"
    manifest.write_text(json.dumps(report()["server"]))
    output = tmp_path / "results.json"
    try:
        run(
            SimpleNamespace(
                requests=fixtures,
                server_manifest=manifest,
                label="test",
                base_url=f"http://127.0.0.1:{server.server_port}",
                model="model",
                scenario="cold",
                concurrency=1,
                timeout=5,
                rounds=3,
                metrics_settle_seconds=0,
                output=output,
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    results = json.loads(output.read_text())
    assert state == {"requests": 4, "resets": 3}
    assert len(results["rounds"]) == 3
    assert all(r["summary"]["successful"] == 1 for r in results["rounds"])
    assert "must-not-persist" not in output.read_text()
