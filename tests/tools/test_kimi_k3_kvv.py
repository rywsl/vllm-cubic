# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline tests for the Kimi Vendor Verifier evidence boundary."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from argparse import Namespace
from pathlib import Path

import httpx
import pytest

TOOLS = Path(__file__).parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import kimi_k3_kvv as runner  # noqa: E402
import kimi_k3_kvv_plugin as plugin  # noqa: E402


def test_lfs_pointer_parser_requires_oid_and_size() -> None:
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:" + b"a" * 64 + b"\nsize 12\n"
    )
    assert runner.parse_lfs_pointer(pointer) == ("a" * 64, 12)
    assert runner.parse_lfs_pointer(b"ordinary data") is None


def test_trace_httpx_mock_records_hash_status_and_request_id(
    tmp_path: Path, monkeypatch
) -> None:
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("KVV_TRACE_FILE", str(trace))
    # A previous test process may have imported the plugin, so this test uses
    # the installed wrapper directly and remains independent of pytest hooks.
    plugin.install_http_trace()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, headers={"x-request-id": "req-local-1"}, json={"error": "bad"}
        )

    body = b'{"model":"local","secret":"do-not-write"}'
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = client.post(
            "https://vendor.example/v1/chat/completions",
            content=body,
            headers={"Authorization": "Bearer sk-test-secret-value"},
        )
    assert response.status_code == 400
    record = json.loads(trace.read_text(encoding="utf-8"))
    assert record["status_code"] == 400
    assert record["request_id"] == "req-local-1"
    assert record["request_body_sha256"] == runner.sha256_bytes(body)
    assert record["authorization"] == "Bearer ***"
    assert "secret-value" not in trace.read_text(encoding="utf-8")
    assert "do-not-write" not in trace.read_text(encoding="utf-8")


def test_redact_text_removes_bearer_and_token_shapes() -> None:
    value = "Bearer sk-live-1234567890123456 token-abcdefghijklmnop"
    cleaned = runner.redact_text(value, "sk-live-1234567890123456")
    assert "1234567890123456" not in cleaned
    assert "Bearer ***" in cleaned
    assert "token-abcdefghijklmnop" not in cleaned


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@vendor.example/v1",
        "https://vendor.example/v1?key=secret",
        "https://vendor.example/v1#secret",
        "ftp://localhost/v1",
        "file://localhost/v1",
        "http://vendor.example/v1",
    ],
)
def test_target_url_rejects_credentials_and_unsupported_schemes(url: str) -> None:
    with pytest.raises(runner.KVVError):
        runner.normalize_base_url(url)


@pytest.mark.parametrize(
    "url",
    ["http://localhost:8000/v1", "http://127.0.0.1:8000/v1", "http://[::1]:8000/v1"],
)
def test_target_url_allows_local_http(url: str) -> None:
    assert runner.normalize_base_url(url + "/") == url


def test_target_model_falls_back_to_official_env_name(monkeypatch) -> None:
    monkeypatch.setenv("KIMI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("KIMI_API_KEY", "local-dummy-key")
    monkeypatch.setenv("MODEL_NAME", "official-fallback")
    monkeypatch.delenv("KIMI_MODEL_NAME", raising=False)
    args = Namespace(base_url="", model="")
    assert runner.target_from_env(args) == (
        "http://localhost:8000/v1",
        "official-fallback",
        "local-dummy-key",
    )
    monkeypatch.setenv("KIMI_MODEL_NAME", "preferred-model")
    assert runner.target_from_env(args)[1] == "preferred-model"


@pytest.mark.parametrize("module", [runner, plugin])
def test_redaction_removes_underscore_token_payload(module) -> None:
    credential = "token_abcdefghijklmnopqrstuvwx"
    assert credential not in module.redact_text(f"api_key={credential}")
    assert "abcdefghijklmnopqrstuvwx" not in module.redact_text(credential)


PARAM_NODE = (
    "tests/params/test_params.py::test_wrong_param_rejected[thinking-top_p=0.8]"
)
FEATURE_NODE = "tests/k3_features/test_response_format.py::test_json_object[stream]"
SCHEMA_NODE = (
    "tests/tool_call_json_schema/test_tool_call_json_schema.py::"
    "test_tool_call_schema_matches_case_schema[TestBasicTypes:1:stream]"
)
PROMPT_NODE = (
    "tests/prompt_tokens/test_prompt_tokens.py::"
    "test_prompt_tokens_match_groundtruth[fixture-a]"
)
ALLOWED_SKIP = (
    "tests/k3_features/test_thinking_effort.py::"
    "test_effort_reasoning_length_increases[stream]"
)


def _bundle(
    tmp_path: Path,
    *,
    feature_node: str = FEATURE_NODE,
    outcomes: dict[str, str] | None = None,
    actual_nodes: dict[str, list[str]] | None = None,
    rejection_status: int | None = 400,
) -> runner.Config:
    """Build complete local pytest artifacts, including both schema stages."""
    config = runner.Config(tmp_path / "source", tmp_path, tmp_path / "venv")
    expected = {
        "params": [PARAM_NODE],
        "k3_features": [feature_node],
        "tool_schema_off": [SCHEMA_NODE],
        "tool_schema_on": [SCHEMA_NODE],
        "prompttokens": [PROMPT_NODE],
    }
    collected = sorted({node for values in expected.values() for node in values})
    runner.write_json(
        tmp_path / "collection.json",
        {"collected": len(collected), "nodeids": collected},
    )
    runner.write_json(
        tmp_path / "manifest.json",
        {
            "source": {"commit": runner.VERIFIER_COMMIT, "clean": True},
            "target": {"model": "local-fixture"},
            "run": {
                "stages": list(expected),
                "pytest_exit_codes": {stage: 0 for stage in expected},
            },
        },
    )
    evidence = tmp_path / "evidence"
    evidence.mkdir(exist_ok=True)
    for stage, expected_nodes in expected.items():
        nodes = (actual_nodes or {}).get(stage, expected_nodes)
        suite: ET.Element = ET.Element(
            "testsuite", name="pytest", tests=str(len(nodes))
        )
        attempts, traces = [], []
        for nodeid in nodes:
            source, name = nodeid.split("::", 1)
            outcome = (outcomes or {}).get(stage, "passed")
            case = ET.SubElement(
                suite,
                "testcase",
                classname=source.removesuffix(".py").replace("/", "."),
                name=name,
                time="0.001",
            )
            detail = None
            if outcome != "passed":
                tag = "failure" if outcome == "failed" else outcome
                detail = (
                    "temporarily skipped: reasoning length is noisy"
                    if nodeid == ALLOWED_SKIP
                    else "local fixture outcome"
                )
                ET.SubElement(case, tag, message=detail).text = detail
            attempts.append(
                {
                    "nodeid": nodeid,
                    "phase": "setup" if outcome == "skipped" else "call",
                    "outcome": outcome,
                    "failure": detail if outcome == "failed" else None,
                    "skip_reason": detail if outcome == "skipped" else None,
                }
            )
            if outcome != "skipped":
                traces.append(
                    {
                        "nodeid": nodeid,
                        "method": "POST",
                        "url": "http://localhost:8000/v1/chat/completions",
                        "request_body_sha256": "a" * 64,
                        "status_code": rejection_status if stage == "params" else 200,
                        "request_id": f"local-{stage}",
                        "authorization": "Bearer ***",
                        "error": "ConnectError" if rejection_status is None else None,
                    }
                )
        ET.ElementTree(suite).write(evidence / f"{stage}.xml", encoding="unicode")
        (evidence / f"{stage}.attempts.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in attempts),
            encoding="utf-8",
        )
        (evidence / f"{stage}.trace.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in traces),
            encoding="utf-8",
        )
    return config


def test_summary_accepts_complete_stages_with_real_rejection_400(
    tmp_path: Path,
) -> None:
    summary = runner.summarize(_bundle(tmp_path))
    assert summary["status"] == "pass"
    assert summary["collection"]["complete"] is True
    assert summary["rejection_http_status"]["ok"] is True
    assert summary["official"]["total"] == 5


@pytest.mark.parametrize(
    "actual_nodes",
    [
        # Running schema-off never substitutes for schema-on.
        {"tool_schema_on": []},
        # The exact test must run in its own stage, even when present elsewhere.
        {"prompttokens": [], "k3_features": [FEATURE_NODE, PROMPT_NODE]},
        # A matching test name in a different source file is not the same test.
        {
            "prompttokens": [
                PROMPT_NODE.replace("test_prompt_tokens.py", "test_other.py")
            ]
        },
        # Parameter suffixes must be preserved when checking completeness.
        {"prompttokens": [PROMPT_NODE.replace("fixture-a", "fixture-b")]},
    ],
)
def test_summary_rejects_missing_or_misplaced_exact_nodeid(
    tmp_path: Path, actual_nodes: dict[str, list[str]]
) -> None:
    summary = runner.summarize(_bundle(tmp_path, actual_nodes=actual_nodes))
    assert summary["status"] != "pass"
    assert summary["collection"]["complete"] is False


def test_summary_failure_cannot_be_a_pass(tmp_path: Path) -> None:
    summary = runner.summarize(_bundle(tmp_path, outcomes={"k3_features": "failed"}))
    assert summary["official"]["failed"] == 1
    assert summary["status"] != "pass"


def test_summary_preserves_only_official_skips(tmp_path: Path) -> None:
    allowed = runner.summarize(
        _bundle(
            tmp_path / "allowed",
            feature_node=ALLOWED_SKIP,
            outcomes={"k3_features": "skipped"},
        )
    )
    assert allowed["status"] == "pass"
    assert allowed["official"]["skipped"] == 1
    assert allowed["skip_allowlist"]["unexpected"] == []
    unexpected = runner.summarize(
        _bundle(tmp_path / "unexpected", outcomes={"k3_features": "skipped"})
    )
    assert unexpected["status"] != "pass"
    assert len(unexpected["skip_allowlist"]["unexpected"]) == 1


@pytest.mark.parametrize("status", [None, 401, 403, 422, 500])
def test_official_rejection_pass_requires_observed_http_400(
    tmp_path: Path, status: int | None
) -> None:
    summary = runner.summarize(_bundle(tmp_path, rejection_status=status))
    assert summary["official"]["failed"] == 0
    assert summary["rejection_http_status"]["ok"] is False
    assert summary["status"] != "pass"


def test_retry_summary_keeps_first_failure_and_final_outcome() -> None:
    attempts = [
        {"nodeid": FEATURE_NODE, "phase": "setup", "outcome": "passed"},
        {
            "nodeid": FEATURE_NODE,
            "phase": "call",
            "outcome": "failed",
            "failure": "first failure evidence",
        },
        {
            "nodeid": FEATURE_NODE,
            "phase": "call",
            "outcome": "failed",
            "failure": "second failure evidence",
        },
        {"nodeid": FEATURE_NODE, "phase": "call", "outcome": "passed"},
    ]
    result = runner.merge_attempts(attempts)
    assert len(result) == 1
    assert result[0]["attempt_count"] == 3
    assert result[0]["retry_count"] == 2
    assert result[0]["first_failure"] == "first failure evidence"
    assert result[0]["final_outcome"] == "passed"


@pytest.mark.parametrize("command", ["run", "summarize"])
def test_cli_returns_nonzero_for_failed_acceptance(
    tmp_path: Path, monkeypatch, command: str
) -> None:
    monkeypatch.setattr(runner, "run_suite", lambda *args: {"status": "fail"})
    monkeypatch.setattr(runner, "summarize", lambda *args: {"status": "fail"})
    assert runner.main([command, "--work-dir", str(tmp_path)]) != 0
