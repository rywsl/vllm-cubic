# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi K3 dynamic-tool request compatibility.

K3 accepts function tools on individual ``system`` messages.  The regular
OpenAI request model only has a top-level ``tools`` field, so this module keeps
the raw message declarations separate and exposes a single effective tool view
to the renderer and parser.  It intentionally has no dependency on the
protocol module at import time; the protocol validator imports it before
Pydantic normalisation.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any, NoReturn

import regex as re

from vllm import envs
from vllm.exceptions import VLLMValidationError

_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,255}$")


def enabled() -> bool:
    """Return whether the opt-in K3 vendor compatibility contract is active."""
    return bool(getattr(envs, "VLLM_KIMI_K3_API_COMPAT", False))


def _error(message: str, parameter: str, value: Any = None) -> NoReturn:
    raise VLLMValidationError(message, parameter=parameter, value=value)


def _tool_name(tool: Mapping[str, Any], parameter: str) -> str:
    if not isinstance(tool, Mapping):
        _error("Each tool must be an object.", parameter, tool)
    if tool.get("type") != "function":
        _error(
            "Only function tools are supported.",
            f"{parameter}.type",
            tool.get("type"),
        )
    function = tool.get("function")
    if not isinstance(function, Mapping):
        _error("Function tool must contain a function object.", f"{parameter}.function")
    name = function.get("name")
    if not isinstance(name, str) or _TOOL_NAME_RE.fullmatch(name) is None:
        _error(
            "Tool function name must start with a letter or '_', contain only "
            "letters, digits, '_' or '-', and be at most 256 characters.",
            f"{parameter}.function.name",
            name,
        )
    if "parameters" not in function:
        _error(
            "Tool function must contain parameters.",
            f"{parameter}.function.parameters",
        )
    if not isinstance(function["parameters"], Mapping):
        _error(
            "Tool function parameters must be an object.",
            f"{parameter}.function.parameters",
            function["parameters"],
        )
    return name


def _iter_tools(value: Any, parameter: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        _error("Tools must be an array.", parameter, value)
    return value


def _message_content_is_empty(message: Mapping[str, Any]) -> bool:
    content = message.get("content")
    if content is None or content == "":
        return True
    return isinstance(content, list) and not content


def _validate_tool_message_links(messages: list[Mapping[str, Any]]) -> None:
    """Validate tool-result ids against the preceding assistant call block."""
    known_ids: set[str] = set()
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                _error(
                    "assistant.tool_calls must be an array.",
                    f"messages[{index}].tool_calls",
                )
            known_ids = {
                str(call.get("id"))
                for call in calls
                if isinstance(call, Mapping) and call.get("id") is not None
            }
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                _error(
                    "Tool messages require a non-empty tool_call_id.",
                    f"messages[{index}].tool_call_id",
                    call_id,
                )
            if call_id not in known_ids:
                _error(
                    "tool_call_id does not match a preceding assistant tool call.",
                    f"messages[{index}].tool_call_id",
                    call_id,
                )


def validate(data: dict[str, Any]) -> dict[str, Any]:
    """Validate raw K3 dynamic tools and return the same payload.

    The returned object is the original mapping so callers can preserve message
    placement and avoid adding dynamic declarations to the top-level template
    field.  When compatibility is disabled this function is a no-op.
    """
    if not enabled() or not isinstance(data, dict):
        return data
    messages = data.get("messages")
    if not isinstance(messages, list):
        return data

    names: set[str] = set()
    top_tools = data.get("tools")
    if top_tools is None:
        top_tools = []
    for position, tool in enumerate(_iter_tools(top_tools, "tools")):
        name = _tool_name(tool, f"tools[{position}]")
        if name in names:
            _error("Duplicate tool function name.", "tools", name)
        names.add(name)

    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            _error("Each message must be an object.", f"messages[{index}]", message)
        dynamic = message.get("tools")
        if dynamic is None:
            continue
        if message.get("role") != "system":
            _error(
                "Dynamic tools are only allowed on system messages.",
                f"messages[{index}].tools",
            )
        if not _message_content_is_empty(message):
            _error(
                "A system message carrying dynamic tools must have empty content.",
                f"messages[{index}].content",
            )
        for tool_position, tool in enumerate(
            _iter_tools(dynamic, f"messages[{index}].tools")
        ):
            name = _tool_name(tool, f"messages[{index}].tools[{tool_position}]")
            if name in names:
                _error(
                    "Duplicate tool function name.",
                    f"messages[{index}].tools",
                    name,
                )
            names.add(name)

    _validate_tool_message_links(messages)
    return data


def dynamic_tool_dicts(
    messages: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return dynamic declarations in their original message order."""
    result: list[dict[str, Any]] = []
    for message in messages or []:
        tools = message.get("tools") if isinstance(message, Mapping) else None
        if isinstance(tools, list):
            result.extend(
                copy.deepcopy(dict(tool)) for tool in tools if isinstance(tool, Mapping)
            )
    return result


def effective_tool_dicts(request_or_data: Any) -> list[dict[str, Any]]:
    """Return top-level tools followed by dynamic tools for parser/grammar use."""
    if isinstance(request_or_data, Mapping):
        top = request_or_data.get("tools") or []
        messages = request_or_data.get("messages") or []
    else:
        top = getattr(request_or_data, "tools", None) or []
        messages = getattr(request_or_data, "messages", None) or []

    result: list[dict[str, Any]] = []
    for tool in top:
        if isinstance(tool, Mapping):
            result.append(copy.deepcopy(dict(tool)))
        else:
            dump = getattr(tool, "model_dump", None)
            result.append(
                dump(mode="json", exclude_none=True) if callable(dump) else dict(tool)
            )
    if enabled():
        result.extend(dynamic_tool_dicts(messages))
    return result


def effective_tool_objects(request: Any) -> list[Any] | None:
    """Return the parser's tools without relocating prompt declarations."""
    top_tools = getattr(request, "tools", None)
    if not enabled() or not hasattr(request, "messages"):
        return top_tools
    dynamic = dynamic_tool_dicts(request.messages)
    if not dynamic:
        return top_tools
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionToolsParam,
    )

    return list(top_tools or []) + [
        ChatCompletionToolsParam.model_validate(tool) for tool in dynamic
    ]


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    """Compatibility alias used by request validators."""
    return validate(data)


# Stable names used by the request protocol.  Keep these wrappers instead of
# assigning aliases so tracebacks and type checkers point at this module's
# public contract.
def validate_kimi_k3_tools(data: dict[str, Any]) -> dict[str, Any]:
    return validate(data)


def get_kimi_k3_tools(request_or_data: Any) -> list[dict[str, Any]] | None:
    """Return available raw tools, preserving ``None`` for no declarations."""
    if isinstance(request_or_data, dict):
        validate(request_or_data)
    result = effective_tool_dicts(request_or_data)
    if result:
        return result
    if isinstance(request_or_data, Mapping):
        if "tools" not in request_or_data or request_or_data.get("tools") is None:
            return None
    elif getattr(request_or_data, "tools", None) is None:
        return None
    return result


def get_effective_tools(request_or_data: Any) -> list[Any] | None:
    return effective_tool_objects(request_or_data)


__all__ = [
    "dynamic_tool_dicts",
    "effective_tool_dicts",
    "effective_tool_objects",
    "enabled",
    "get_effective_tools",
    "get_kimi_k3_tools",
    "normalize",
    "validate",
    "validate_kimi_k3_tools",
]
