# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pytest support for the Kimi Vendor Verifier acceptance runner.

The verifier is intentionally kept as an unmodified, pinned checkout.  This
plugin adds only evidence collection around it: request metadata is recorded
without request bodies, and test attempts are recorded before the runner
summarises JUnit output.  It is safe to load this module for collection-only
invocations.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import regex as re

_NODEID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "kvv_nodeid", default=None
)
_PATCH_LOCK = threading.Lock()

_TOKEN_RE = re.compile(r"(?i)\b((?:sk|key|token)[-_])[A-Za-z0-9._~-]{12,}")
_BEARER_RE = re.compile(r"(?i)(\bBearer\s+)(?!\*{3}\b)[A-Za-z0-9._~+/-]{8,}")


def redact_text(value: str, secret: str | None = None) -> str:
    """Remove credentials from text before it is persisted."""
    secret = secret or os.environ.get("KIMI_API_KEY")
    if secret:
        value = value.replace(secret, "***")
    value = _BEARER_RE.sub(r"\1***", value)
    return _TOKEN_RE.sub(r"\1***", value)


def _path(name: str, fallback: str) -> Path | None:
    value = os.environ.get(name, fallback)
    return Path(value).expanduser().resolve() if value else None


def _worker_path(path: Path) -> Path:
    """Give each xdist worker an independent append-only evidence file."""
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker:
        return path
    return path.with_name(f"{path.name}.{worker}")


def _append_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with _PATCH_LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(line)


def _request_body(request: Any) -> bytes | None:
    try:
        body = request.content
    except Exception:
        return None
    return body if isinstance(body, bytes) else None


def _safe_url(request: Any) -> str:
    try:
        url = request.url
        return f"{url.scheme}://{url.host}{url.path}"
    except Exception:
        return "<unknown>"


def _record_http(
    request: Any,
    *,
    response: Any | None = None,
    error: BaseException | None = None,
) -> None:
    path = _path("KVV_TRACE_FILE", "")
    if path is None:
        return
    body = _request_body(request)
    headers = getattr(response, "headers", {}) if response is not None else {}
    request_id = None
    for name in ("x-request-id", "request-id", "x-correlation-id"):
        request_id = headers.get(name)
        if request_id:
            break
    record: dict[str, Any] = {
        "nodeid": _NODEID.get(),
        "method": getattr(request, "method", None),
        "url": _safe_url(request),
        "request_body_sha256": hashlib.sha256(body).hexdigest()
        if body is not None
        else None,
        "request_body_bytes": len(body) if body is not None else None,
        "authorization": "Bearer ***" if "authorization" in request.headers else None,
        "status_code": getattr(response, "status_code", None),
        "request_id": request_id,
        "error": type(error).__name__ if error is not None else None,
    }
    _append_json(_worker_path(path), record)


def install_http_trace() -> None:
    """Patch httpx clients once, recording status and request hashes."""
    try:
        import httpx
    except ImportError:
        return
    if getattr(httpx.Client, "_kvv_trace_installed", False):
        return

    original_send = httpx.Client.send

    def send(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            response = original_send(self, request, *args, **kwargs)
        except BaseException as exc:
            _record_http(request, error=exc)
            raise
        _record_http(request, response=response)
        return response

    httpx.Client.send = send  # type: ignore[method-assign]
    httpx.Client._kvv_trace_installed = True  # type: ignore[attr-defined]

    # The verifier currently uses the synchronous OpenAI client.  Patching the
    # async client as well costs little and prevents an untraced future test.
    original_async_send = httpx.AsyncClient.send

    async def async_send(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            response = await original_async_send(self, request, *args, **kwargs)
        except BaseException as exc:
            _record_http(request, error=exc)
            raise
        _record_http(request, response=response)
        return response

    httpx.AsyncClient.send = async_send  # type: ignore[method-assign]


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--kvv-collection-file",
        default=os.environ.get("KVV_COLLECTION_FILE", ""),
        help="Write collected node IDs to this JSON file.",
    )
    parser.addoption(
        "--kvv-attempts-file",
        default=os.environ.get("KVV_ATTEMPTS_FILE", ""),
        help="Write one JSON record per test call attempt.",
    )


def pytest_configure(config: Any) -> None:
    install_http_trace()


def pytest_runtest_call(item: Any) -> Any:
    token = _NODEID.set(item.nodeid)
    try:
        yield
    finally:
        _NODEID.reset(token)


pytest_runtest_call.hookwrapper = True  # type: ignore[attr-defined]


def pytest_collection_finish(session: Any) -> None:
    value = session.config.getoption("kvv_collection_file", default="")
    if not value:
        return
    path = Path(value).expanduser().resolve()
    record = {
        "pytest_version": getattr(session.config, "_pytestversion", None),
        "collected": len(session.items),
        "nodeids": [item.nodeid for item in session.items],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _report_reason(report: Any) -> str | None:
    if report.outcome != "skipped":
        return None
    longrepr = getattr(report, "longrepr", None)
    if longrepr is None:
        return "skipped"
    return redact_text(str(longrepr))[-1200:]


def pytest_runtest_makereport(item: Any, call: Any) -> Any:
    outcome = yield
    report = outcome.get_result()
    if call.when not in {"setup", "call"}:
        return
    if report.outcome not in {"failed", "passed", "skipped"}:
        return
    value = item.config.getoption("kvv_attempts_file", default="")
    if not value:
        return
    record = {
        "nodeid": item.nodeid,
        "phase": call.when,
        "outcome": report.outcome,
        "attempt": getattr(report, "count", None),
        "wasxfail": bool(getattr(report, "wasxfail", False)),
        "skip_reason": _report_reason(report),
    }
    if report.outcome == "failed":
        record["failure"] = redact_text(str(getattr(report, "longrepr", "")))[:4000]
    _append_json(_worker_path(Path(value).expanduser().resolve()), record)


pytest_runtest_makereport.hookwrapper = True  # type: ignore[attr-defined]
