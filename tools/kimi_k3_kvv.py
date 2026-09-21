#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare and run the pinned Kimi Vendor Verifier acceptance suite.

The verifier checkout remains an upstream artifact.  This utility owns only
the reproducibility boundary around it: LFS hydration, an isolated uv
environment, collection checks, redacted pytest evidence, and summary data.
It deliberately does not make endpoint requests during ``prepare`` or
``check``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import regex as re
import tomllib

VERIFIER_REPOSITORY = "https://github.com/MoonshotAI/Kimi-Vendor-Verifier"
VERIFIER_COMMIT = "66092cf444c97356c0e11c5078c67116390615d9"
DEFAULT_MODEL = "Kimi-K3-Cubic-2.5Bit"
DEFAULT_WORK_DIR_NAME = "artifacts/kimi-k3-kvv"
PYTEST_PLUGIN = Path(__file__).with_name("kimi_k3_kvv_plugin.py")

STAGES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("params", ("tests/params",), ()),
    ("k3_features", ("tests/k3_features",), ()),
    ("tool_schema_off", ("tests/tool_call_json_schema",), ()),
    ("tool_schema_on", ("tests/tool_call_json_schema",), ("--thinking",)),
    ("prompttokens", ("tests/prompt_tokens",), ()),
)

LFS_POINTER = re.compile(
    rb"version https://git-lfs.github.com/spec/v1\s+"
    rb"oid sha256:([0-9a-f]{64})\s+size ([0-9]+)\s*$"
)
TOKEN_RE = re.compile(r"(?i)\b((?:sk|key|token)[-_])[A-Za-z0-9._~-]{12,}")
BEARER_RE = re.compile(r"(?i)(\bBearer\s+)(?!\*{3}\b)[A-Za-z0-9._~+/-]{8,}")

# These skips are present in the pinned source and are part of the official
# suite.  Anything else is surfaced as an unexpected skip in summarize.
SKIP_ALLOWLIST = {
    "tests/k3_features/test_thinking_effort.py::test_effort_reasoning_length_increases",
    "tests/k3_features/test_thinking_effort.py::test_effort_default_closest_to_max",
    "tests/k3_features/test_thinking_effort.py::test_effort_with_disabled_returns_no_reasoning",
    "tests/k3_features/test_thinking_effort.py::test_disabled_ignores_keep",
    "tests/k3_features/test_thinking_effort.py::test_keep_not_all_rejected",
    "tests/k3_features/test_thinking_effort.py::test_keep_all_preserves_history_reasoning",
    "tests/k3_features/test_thinking_effort.py::test_keep_all_prompt_tokens_include_reasoning",
    "tests/k3_features/test_tool_choice.py::test_tool_choice_named_function_rejected",
    "tests/k3_features/test_tokenization_groundtruth.py::test_prompt_tokens_match_groundtruth",
}


class KVVError(RuntimeError):
    """A setup or evidence error, distinct from an assertion finding."""


@dataclass(frozen=True)
class Config:
    verifier_dir: Path
    work_dir: Path
    venv_dir: Path
    repository: str = VERIFIER_REPOSITORY
    commit: str = VERIFIER_COMMIT


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KVVError(f"invalid JSON evidence {path}: {exc}") from exc


def redact_file(path: Path, secret: str | None = None) -> None:
    """Redact a persisted text artifact in place before it is retained."""
    if not path.is_file():
        return
    try:
        value = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise KVVError(f"evidence is not UTF-8 text: {path}") from exc
    path.write_text(redact_text(value, secret), encoding="utf-8")


def redact_text(value: str, secret: str | None = None) -> str:
    """Redact exact and token-shaped credentials from persisted text."""
    if secret:
        value = value.replace(secret, "***")
    value = BEARER_RE.sub(r"\1***", value)
    return TOKEN_RE.sub(r"\1***", value)


def mask_key(value: str) -> str:
    if not value:
        return "(unset)"
    return f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"


def normalize_base_url(value: str) -> str:
    value = value.rstrip("/")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError as exc:
        raise KVVError("base URL is malformed") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise KVVError("base URL must be an absolute URL without query or fragment")
    if parsed.scheme != "https" and hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise KVVError("non-local base URL must use HTTPS")
    return value


def git_output(config: Config, *args: str, env: dict[str, str] | None = None) -> str:
    command = ["git", *args]
    result = subprocess.run(
        command,
        cwd=config.verifier_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise KVVError(f"git {' '.join(args)} failed: {redact_text(result.stderr)}")
    return result.stdout.strip()


def run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
    secret: str | None = None,
    allowed: set[int] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a child and persist only redacted combined output."""
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output = redact_text((result.stdout or "") + (result.stderr or ""), secret)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
    accepted = allowed or {0}
    if result.returncode not in accepted:
        tail = "\n".join(output.splitlines()[-20:])
        raise KVVError(
            f"command exited {result.returncode}: {' '.join(map(str, command))}\n{tail}"
        )
    return subprocess.CompletedProcess(result.args, result.returncode, output, "")


def parse_lfs_pointer(data: bytes) -> tuple[str, int] | None:
    match = LFS_POINTER.fullmatch(data.strip())
    if match is None:
        return None
    return match.group(1).decode("ascii"), int(match.group(2))


def lfs_expected(config: Config) -> dict[str, dict[str, Any]]:
    paths = git_output(
        config, "ls-tree", "-r", "--name-only", config.commit
    ).splitlines()
    expected: dict[str, dict[str, Any]] = {}
    for relative in paths:
        try:
            pointer = subprocess.run(
                ["git", "show", f"{config.commit}:{relative}"],
                cwd=config.verifier_dir,
                capture_output=True,
                check=True,
            ).stdout
        except subprocess.CalledProcessError:
            continue
        parsed = parse_lfs_pointer(pointer)
        if parsed is not None:
            oid, size = parsed
            expected[relative] = {"oid": oid, "size": size}
    return expected


def verify_source(config: Config, *, check_status: bool = True) -> dict[str, Any]:
    if not config.verifier_dir.is_dir():
        raise KVVError(f"verifier directory does not exist: {config.verifier_dir}")
    actual = git_output(config, "rev-parse", "HEAD")
    if actual != config.commit:
        raise KVVError(f"verifier HEAD {actual} does not match pinned {config.commit}")
    origin = git_output(config, "config", "--get", "remote.origin.url")
    if origin.rstrip("/") != config.repository.rstrip("/"):
        raise KVVError(
            f"verifier origin {origin!r} does not match {config.repository!r}"
        )
    clean = True
    if check_status:
        status = git_output(config, "status", "--porcelain")
        if status:
            raise KVVError(f"verifier checkout is dirty:\n{status}")
        clean = True
    return {"repository": config.repository, "commit": actual, "clean": clean}


def verify_lfs(config: Config) -> list[dict[str, Any]]:
    expected = lfs_expected(config)
    if not expected:
        raise KVVError("pinned verifier contains no discoverable LFS pointers")
    records: list[dict[str, Any]] = []
    for relative, item in sorted(expected.items()):
        path = config.verifier_dir / relative
        if not path.is_file():
            raise KVVError(f"missing LFS file: {relative}")
        size = path.stat().st_size
        digest = sha256_file(path)
        hydrated = not parse_lfs_pointer(path.read_bytes()[:512])
        record = {
            "path": relative,
            "expected_oid": item["oid"],
            "actual_oid": digest,
            "expected_size": item["size"],
            "actual_size": size,
            "hydrated": hydrated,
            "ok": hydrated and digest == item["oid"] and size == item["size"],
        }
        records.append(record)
        if not record["ok"]:
            raise KVVError(f"LFS verification failed for {relative}: {record}")
    return records


def find_git_lfs(config: Config) -> tuple[str, dict[str, str]]:
    """Return a git-lfs command and PATH with no system configuration changes."""
    existing = shutil.which("git-lfs")
    if existing:
        return existing, os.environ.copy()
    probe = subprocess.run(
        ["git", "lfs", "version"], capture_output=True, text=True, check=False
    )
    if probe.returncode == 0:
        return "git-lfs", os.environ.copy()

    tool_dir = config.work_dir / "tooling" / "git-lfs"
    tool_dir.mkdir(parents=True, exist_ok=True)
    version = os.environ.get("KVV_GIT_LFS_VERSION", "3.6.1")
    archive = tool_dir / f"git-lfs-linux-amd64-v{version}.tar.gz"
    if not archive.is_file():
        url = f"https://github.com/git-lfs/git-lfs/releases/download/v{version}/{archive.name}"
        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                archive.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream)
        except Exception as exc:
            raise KVVError(
                f"unable to download temporary git-lfs {version}: {exc}"
            ) from exc
    with tarfile.open(archive, "r:gz") as bundle:
        root = tool_dir.resolve()
        for member in bundle.getmembers():
            target = (tool_dir / member.name).resolve()
            if target != root and root not in target.parents:
                raise KVVError(
                    f"git-lfs archive contains an unsafe path: {member.name}"
                )
        bundle.extractall(tool_dir, filter="data")
    executable = next(tool_dir.glob(f"git-lfs-{version}/git-lfs"), None)
    if executable is None:
        raise KVVError(
            f"git-lfs archive did not contain expected executable: {archive}"
        )
    executable.chmod(0o755)
    environment = os.environ.copy()
    environment["PATH"] = (
        f"{executable.parent}{os.pathsep}{environment.get('PATH', '')}"
    )
    return str(executable), environment


def dependency_specs(pyproject: Path) -> list[str]:
    """Read the verifier's dependency declarations without modifying pyproject."""
    with pyproject.open("rb") as stream:
        document = tomllib.load(stream)
    specs = list(document.get("project", {}).get("dependencies", []))
    if not specs:
        raise KVVError(f"could not read dependencies from {pyproject}")
    return specs


def prepare(config: Config, *, install: bool = True) -> dict[str, Any]:
    config.work_dir.mkdir(parents=True, exist_ok=True)
    source = verify_source(config, check_status=False)
    lfs_command, lfs_env = find_git_lfs(config)
    lfs_env["GIT_LFS_SKIP_SMUDGE"] = "0"
    run_process(
        [lfs_command, "install", "--local", "--force"],
        cwd=config.verifier_dir,
        env=lfs_env,
    )
    lfs_executable = str(
        Path(
            shutil.which(lfs_command, path=lfs_env.get("PATH")) or lfs_command
        ).resolve()
    )
    for key, value in {
        "filter.lfs.clean": f"{lfs_executable} clean -- %f",
        "filter.lfs.smudge": f"{lfs_executable} smudge -- %f",
        "filter.lfs.process": f"{lfs_executable} filter-process",
    }.items():
        run_process(
            ["git", "config", "--local", key, value],
            cwd=config.verifier_dir,
            env=lfs_env,
        )
    run_process(
        [lfs_command, "pull"],
        cwd=config.verifier_dir,
        env=lfs_env,
        log_path=config.work_dir / "git-lfs.log",
    )
    source = verify_source(config)
    records = verify_lfs(config)

    python = config.venv_dir / "bin" / "python"
    if install:
        config.venv_dir.parent.mkdir(parents=True, exist_ok=True)
        if not python.is_file():
            run_process(
                ["uv", "venv", "--python", "3.12", str(config.venv_dir)],
                cwd=config.verifier_dir,
                log_path=config.work_dir / "uv-venv.log",
            )
        specs = dependency_specs(config.verifier_dir / "pyproject.toml")
        run_process(
            ["uv", "pip", "install", "--python", str(python), *specs],
            cwd=config.verifier_dir,
            log_path=config.work_dir / "uv-install.log",
        )
    if not python.is_file():
        raise KVVError(f"verifier Python is missing: {python}")
    versions = dependency_versions(python, config.verifier_dir)
    result = {
        "generated_at": now_iso(),
        "source": source,
        "lfs": records,
        "venv": str(config.venv_dir),
        "dependencies": versions,
    }
    write_json(config.work_dir / "prepare.json", result)
    return result


def dependency_versions(python: Path, cwd: Path) -> dict[str, str]:
    code = """
import importlib.metadata as metadata
import json
import platform

versions = {"python": platform.python_version()}
for name in ("pytest", "pytest-rerunfailures", "pytest-xdist", "httpx", "openai"):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        pass
print(json.dumps(versions))
"""
    result = subprocess.run(
        [str(python), "-c", code], cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise KVVError(
            f"unable to inspect verifier environment: {redact_text(result.stderr)}"
        )
    return json.loads(result.stdout)


def collection_command(config: Config, output: Path) -> list[str]:
    return [
        str(config.venv_dir / "bin" / "python"),
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "tests",
        "-p",
        "kimi_k3_kvv_plugin",
        "--kvv-collection-file",
        str(output),
    ]


def plugin_env() -> dict[str, str]:
    environment = os.environ.copy()
    root = str(PYTEST_PLUGIN.parent)
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = root + (os.pathsep + previous if previous else "")
    return environment


def check(config: Config, *, collect: bool = True) -> dict[str, Any]:
    source = verify_source(config)
    lfs = verify_lfs(config)
    python = config.venv_dir / "bin" / "python"
    if not python.is_file():
        raise KVVError(f"verifier Python is missing: {python}")
    collection: dict[str, Any] | None = None
    if collect:
        path = config.work_dir / "collection.json"
        run_process(
            collection_command(config, path),
            cwd=config.verifier_dir,
            env=plugin_env(),
            log_path=config.work_dir / "collect.log",
        )
        collection = read_json(path)
        if not collection.get("nodeids") or collection.get("collected") != len(
            collection["nodeids"]
        ):
            raise KVVError("collection evidence is empty or inconsistent")
    result = {
        "generated_at": now_iso(),
        "source": source,
        "lfs": lfs,
        "python": str(python),
        "collection": collection,
        "ok": True,
    }
    write_json(config.work_dir / "check.json", result)
    return result


def target_from_env(args: argparse.Namespace) -> tuple[str, str, str]:
    base_url = normalize_base_url(args.base_url or os.environ.get("KIMI_BASE_URL", ""))
    model = (
        args.model
        or os.environ.get("KIMI_MODEL_NAME")
        or os.environ.get("MODEL_NAME", "")
    )
    key = os.environ.get("KIMI_API_KEY", "")
    if not model:
        raise KVVError("MODEL_NAME or KIMI_MODEL_NAME is required for run")
    if not key:
        raise KVVError(
            "KIMI_API_KEY is required for run and is never accepted as a CLI argument"
        )
    return base_url, model, key


def run_suite(config: Config, args: argparse.Namespace) -> dict[str, Any]:
    evidence = config.work_dir / "evidence"
    if evidence.exists() and any(evidence.iterdir()):
        raise KVVError(
            f"refusing to append to existing evidence directory: {evidence}; "
            "choose a new --work-dir"
        )
    base_url, model, key = target_from_env(args)
    check_record = check(config, collect=True)
    collection = check_record["collection"]
    manifest = {
        "generated_at": now_iso(),
        "target": {
            "base_url": base_url,
            "model": model,
            "api_key_env": "KIMI_API_KEY",
            "api_key_masked": mask_key(key),
            "authorization": "Bearer ***",
        },
        "source": check_record["source"],
        "run": {
            "concurrency": args.concurrency,
            "think_mode": args.think_mode,
            "stages": [name for name, _, _ in STAGES],
            "pytest_exit_codes": {},
            "collection_count": collection["collected"],
        },
    }
    write_json(config.work_dir / "manifest.json", manifest)
    logs = config.work_dir / "logs"
    evidence.mkdir(parents=True, exist_ok=True)
    environment = plugin_env()
    environment.update(
        {
            "KIMI_BASE_URL": base_url,
            "KIMI_API_KEY": key,
            "MODEL_NAME": model,
            "KIMI_MODEL_NAME": model,
            "THINK_MODE": args.think_mode,
        }
    )
    for name, tests, extra in STAGES:
        trace = evidence / f"{name}.trace.jsonl"
        attempts = evidence / f"{name}.attempts.jsonl"
        junit = evidence / f"{name}.xml"
        tool_report = evidence / f"{name}.tool-report.json"
        tool_report_args = (
            ["--tool-json-report", str(tool_report)]
            if name.startswith("tool_schema")
            else []
        )
        command = [
            str(config.venv_dir / "bin" / "python"),
            "-m",
            "pytest",
            "-q",
            "-ra",
            "-n",
            str(args.concurrency),
            *tests,
            "--base-url",
            base_url,
            "--smoke-model",
            model,
            "--think-mode",
            args.think_mode,
            "--junitxml",
            str(junit),
            *tool_report_args,
            "-p",
            "kimi_k3_kvv_plugin",
            "--kvv-attempts-file",
            str(attempts),
            *extra,
        ]
        environment["KVV_TRACE_FILE"] = str(trace)
        environment["KVV_ATTEMPTS_FILE"] = str(attempts)
        result = run_process(
            command,
            cwd=config.verifier_dir,
            env=environment,
            log_path=logs / f"{name}.log",
            secret=key,
            allowed={0, 1},
        )
        merge_worker_attempts(attempts)
        merge_worker_jsonl(trace)
        redact_file(junit, key)
        redact_file(tool_report, key)
        manifest["run"]["pytest_exit_codes"][name] = result.returncode
        write_json(config.work_dir / "manifest.json", manifest)
    summary = summarize(config)
    write_json(config.work_dir / "summary.json", summary)
    return summary


def junit_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        raise KVVError(f"missing JUnit evidence: {path}")
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise KVVError(f"invalid JUnit evidence {path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    embedded_reruns = 0
    for case in root.iter("testcase"):
        status = "passed"
        detail = ""
        for child in list(case):
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "rerun" or tag.startswith("flaky"):
                embedded_reruns += 1
            if tag in {"failure", "error", "skipped"} and status == "passed":
                status = "failed" if tag == "failure" else tag
                detail = (
                    (child.get("message") or "") + "\n" + (child.text or "")
                ).strip()[:4000]
        classname = case.get("classname", "")
        source = classname if "/" in classname else classname.replace(".", "/")
        if not source.endswith(".py"):
            source += ".py"
        records.append(
            {
                "classname": classname,
                "name": case.get("name", ""),
                "nodeid": f"{source}::{case.get('name', '')}",
                "status": status,
                "detail": redact_text(detail),
                "time": float(case.get("time", "0") or 0),
            }
        )
    return records, embedded_reruns


def load_attempts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise KVVError(
                f"invalid attempt evidence {path}:{line_number}: {exc}"
            ) from exc
        records.append(value)
    return records


def merge_worker_jsonl(path: Path) -> None:
    """Merge and remove per-worker JSONL evidence emitted by the plugin."""
    worker_paths = sorted(path.parent.glob(f"{path.name}.gw*"))
    if not worker_paths:
        return
    with path.open("a", encoding="utf-8") as destination:
        for worker_path in worker_paths:
            destination.write(worker_path.read_text(encoding="utf-8"))
            worker_path.unlink()


def merge_worker_attempts(path: Path) -> None:
    """Compatibility wrapper for callers that merge test attempts."""
    merge_worker_jsonl(path)


def merge_attempts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("nodeid", "")), []).append(record)
    merged: list[dict[str, Any]] = []
    for nodeid, attempts in sorted(grouped.items()):
        calls = [item for item in attempts if item.get("phase") == "call"] or attempts
        first_failure = next(
            (item.get("failure") for item in calls if item.get("outcome") == "failed"),
            None,
        )
        final = calls[-1]
        merged.append(
            {
                "nodeid": nodeid,
                "attempt_count": len(calls),
                "retry_count": max(0, len(calls) - 1),
                "first_failure": first_failure,
                "final_outcome": final.get("outcome"),
                "skip_reason": final.get("skip_reason"),
            }
        )
    return merged


def expected_rejection(nodeid: str) -> bool:
    lowered = nodeid.lower()
    return any(
        token in lowered for token in ("rejected", "invalid", "missing_required")
    )


def summarize(config: Config) -> dict[str, Any]:
    manifest_path = config.work_dir / "manifest.json"
    if not manifest_path.is_file():
        raise KVVError(f"missing manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    check_record = (
        read_json(config.work_dir / "check.json")
        if (config.work_dir / "check.json").is_file()
        else None
    )
    collection = read_json(config.work_dir / "collection.json")
    expected_nodeids = set(collection.get("nodeids", []))
    stages: dict[str, Any] = {}
    all_junit: list[dict[str, Any]] = []
    all_attempts: list[dict[str, Any]] = []
    for name, _, _ in STAGES:
        records, embedded = junit_records(config.work_dir / "evidence" / f"{name}.xml")
        attempts = load_attempts(
            config.work_dir / "evidence" / f"{name}.attempts.jsonl"
        )
        merged = merge_attempts(attempts)
        counts = Counter(record["status"] for record in records)
        stages[name] = {
            "total": len(records),
            "counts": dict(sorted(counts.items())),
            "embedded_reruns": embedded,
            "attempts": len(attempts),
            "results": records,
            "attempt_summary": merged,
        }
        all_junit.extend(records)
        all_attempts.extend(attempts)

    actual_nodeids = {record["nodeid"] for record in all_junit}
    missing = sorted(expected_nodeids - actual_nodeids)
    stage_missing: dict[str, list[str]] = {}
    stage_unexpected: dict[str, list[str]] = {}
    for name, test_paths, _ in STAGES:
        expected_stage = {
            nodeid
            for nodeid in expected_nodeids
            if any(nodeid.startswith(f"{path.rstrip('/')}/") for path in test_paths)
        }
        actual_stage = {record["nodeid"] for record in stages[name]["results"]}
        if expected_stage - actual_stage:
            stage_missing[name] = sorted(expected_stage - actual_stage)
        if actual_stage - expected_stage:
            stage_unexpected[name] = sorted(actual_stage - expected_stage)
    unexpected_skips: list[dict[str, Any]] = []
    for record in all_junit:
        if record["status"] == "skipped":
            base = record["nodeid"].split("[", 1)[0]
            if base not in SKIP_ALLOWLIST:
                unexpected_skips.append(
                    {"nodeid": record["nodeid"], "detail": record["detail"]}
                )

    trace_records: list[dict[str, Any]] = []
    for name, _, _ in STAGES:
        path = config.work_dir / "evidence" / f"{name}.trace.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                trace_records.append(json.loads(line))
    rejection_records = [
        record for record in all_junit if expected_rejection(record["nodeid"])
    ]
    statuses_by_node: dict[str, list[int | None]] = {}
    for record in trace_records:
        statuses_by_node.setdefault(str(record.get("nodeid")), []).append(
            record.get("status_code")
        )
    rejection_http = {
        nodeid: statuses_by_node.get(nodeid, [])
        for nodeid in {record["nodeid"] for record in rejection_records}
        if nodeid in statuses_by_node
    }
    missing_rejection_http = sorted(
        nodeid
        for nodeid in {record["nodeid"] for record in rejection_records}
        if 400 not in rejection_http.get(nodeid, [])
    )
    counts = Counter(record["status"] for record in all_junit)
    operational_errors = counts.get("error", 0)
    result = {
        "generated_at": now_iso(),
        "target": manifest.get("target", {}),
        "source": manifest.get("source", {}),
        "run": manifest.get("run", {}),
        "check": check_record,
        "collection": {
            "expected": len(expected_nodeids),
            "executed_junit": len(actual_nodeids),
            "missing": missing,
            "stage_missing": stage_missing,
            "stage_unexpected": stage_unexpected,
            "complete": not missing and not stage_missing and not stage_unexpected,
        },
        "official": {
            "total": len(all_junit),
            "passed": counts.get("passed", 0),
            "failed": counts.get("failed", 0),
            "skipped": counts.get("skipped", 0),
            "errors": operational_errors,
            "pass_rate": round(
                counts.get("passed", 0)
                / (counts.get("passed", 0) + counts.get("failed", 0))
                * 100,
                2,
            )
            if counts.get("passed", 0) + counts.get("failed", 0)
            else None,
            "stages": stages,
        },
        "retries": {
            "attempts": len(all_attempts),
            "by_test": merge_attempts(all_attempts),
        },
        "skip_allowlist": {
            "allowed": sorted(SKIP_ALLOWLIST),
            "unexpected": unexpected_skips,
        },
        "rejection_http_status": {
            "checked": len(rejection_records),
            "statuses": rejection_http,
            "missing_actual_400": missing_rejection_http,
            "ok": not missing_rejection_http,
        },
        "trace": {
            "requests": len(trace_records),
            "status_codes": dict(
                sorted(
                    Counter(
                        str(item.get("status_code")) for item in trace_records
                    ).items()
                )
            ),
            "request_ids": sorted(
                {
                    item.get("request_id")
                    for item in trace_records
                    if item.get("request_id")
                }
            ),
            "records": trace_records,
        },
        "status": "pass"
        if not missing
        and not stage_missing
        and not stage_unexpected
        and not unexpected_skips
        and not counts.get("failed", 0)
        and not operational_errors
        and not missing_rejection_http
        else "fail",
    }
    return result


def config_from_args(args: argparse.Namespace) -> Config:
    script_root = Path(__file__).resolve().parents[1]
    verifier = (
        Path(args.verifier_dir or script_root.parent / "Kimi-Vendor-Verifier")
        .expanduser()
        .resolve()
    )
    work = (
        Path(args.work_dir or script_root / DEFAULT_WORK_DIR_NAME)
        .expanduser()
        .resolve()
    )
    venv = Path(args.venv_dir or verifier / ".venv").expanduser().resolve()
    return Config(verifier_dir=verifier, work_dir=work, venv_dir=venv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_paths(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--verifier-dir", help="Pinned Kimi-Vendor-Verifier checkout"
        )
        command.add_argument("--work-dir", help="Evidence and preparation directory")
        command.add_argument("--venv-dir", help="Separate verifier uv virtualenv")

    prepare_parser = sub.add_parser(
        "prepare", help="hydrate LFS and install verifier dependencies"
    )
    add_paths(prepare_parser)
    prepare_parser.add_argument(
        "--no-install", action="store_true", help="only hydrate and verify LFS"
    )

    check_parser = sub.add_parser(
        "check", help="verify source, LFS, and full collection"
    )
    add_paths(check_parser)
    check_parser.add_argument("--skip-collect", action="store_true")

    run_parser = sub.add_parser(
        "run", help="run the official suite against an endpoint"
    )
    add_paths(run_parser)
    run_parser.add_argument("--base-url", default="", help="defaults to KIMI_BASE_URL")
    run_parser.add_argument("--model", default="", help="defaults to KIMI_MODEL_NAME")
    run_parser.add_argument("--concurrency", type=int, default=4)
    run_parser.add_argument(
        "--think-mode", choices=("opensource", "kimi", "none"), default="opensource"
    )

    summarize_parser = sub.add_parser(
        "summarize", help="summarize sanitized pytest evidence"
    )
    add_paths(summarize_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        if args.command == "prepare":
            result = prepare(config, install=not args.no_install)
        elif args.command == "check":
            result = check(config, collect=not args.skip_collect)
        elif args.command == "run":
            if args.concurrency < 1:
                raise KVVError("--concurrency must be positive")
            result = run_suite(config, args)
        else:
            result = summarize(config)
            write_json(config.work_dir / "summary.json", result)
        print(
            json.dumps(
                {
                    "status": result.get("status", "ok"),
                    "work_dir": str(config.work_dir),
                },
                ensure_ascii=False,
            )
        )
        return 1 if result.get("status") == "fail" else 0
    except KVVError as exc:
        print(f"kimi-k3-kvv: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
