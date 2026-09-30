# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in request compatibility rules for Kimi K3's chat API."""

from __future__ import annotations

import math
from typing import Any

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError

_EFFORTS = ("low", "high", "max")
_THINKING_KEYS = frozenset(("type", "keep", "effort"))
_KEEP_VALUES = ("all", "history")

# A required tool call must leave enough output budget for the XTML tool
# channel.  K3's reasoning can otherwise consume a short request's entire
# ``max_tokens`` before it reaches the call marker.
KIMI_K3_REQUIRED_TOOL_THINKING_TOKEN_BUDGET = 256
KIMI_K3_EFFORT_EXPLICIT_KEY = "_kimi_k3_effort_explicit"


def kimi_k3_api_compat_enabled() -> bool:
    return bool(envs.VLLM_KIMI_K3_API_COMPAT)


def _bad(parameter: str, message: str, value: Any = None) -> VLLMValidationError:
    return VLLMValidationError(message, parameter=parameter, value=value)


def kimi_k3_thinking_token_budget(
    request: Any, current_budget: int | None
) -> int | None:
    """Apply K3's bounded budget for implicit required-tool reasoning.

    The budget is deliberately applied at sampling-parameter construction so
    the regular ``thinking_token_budget`` engine path enforces the limit. An
    explicitly supplied budget or effort always takes precedence.
    """
    if (
        current_budget is not None
        or "thinking_token_budget" in getattr(request, "model_fields_set", set())
        or not kimi_k3_api_compat_enabled()
    ):
        return current_budget
    if getattr(request, "tool_choice", None) != "required":
        return current_budget

    thinking = getattr(request, "thinking", None)
    if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
        return current_budget
    if getattr(request, "_kimi_k3_effort_explicit", False):
        return current_budget

    # Include tools declared on K3 system messages as well as top-level tools.
    from vllm.entrypoints.openai.chat_completion.kimi_k3_tools import (
        effective_tool_objects,
    )

    if not effective_tool_objects(request):
        return current_budget
    return KIMI_K3_REQUIRED_TOOL_THINKING_TOKEN_BUDGET


def _validate_number(data: dict[str, Any], name: str, *, default: float) -> None:
    if name not in data:
        data[name] = default
        return
    value = data[name]
    # bool is an int subclass but is never a valid sampling number here.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bad(name, f"`{name}` must be a number.", value)
    if not math.isfinite(float(value)):
        raise _bad(name, f"`{name}` must be finite.", value)
    if name == "temperature" and not 0 <= value <= 1:
        raise _bad(name, "`temperature` must be between 0 and 1.", value)
    if name == "top_p" and not 0 < value <= 1:
        raise _bad(name, "`top_p` must be greater than 0 and at most 1.", value)
    if name == "top_p" and value != 0.95:
        raise _bad(name, "Kimi K3 requires top_p=0.95.", value)
    if name in ("presence_penalty", "frequency_penalty") and value != 0:
        raise _bad(name, f"Kimi K3 requires {name}=0.", value)


def _validate_integer(data: dict[str, Any], name: str, *, default: int) -> None:
    if name not in data:
        data[name] = default
        return
    value = data[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise _bad(name, f"`{name}` must be an integer.", value)
    if value != 1:
        raise _bad(name, "Kimi K3 requires n=1.", value)


def _validate_response_format(data: dict[str, Any]) -> None:
    response_format = data.get("response_format")
    if response_format is None:
        return
    if not isinstance(response_format, dict):
        raise _bad(
            "response_format", "`response_format` must be an object.", response_format
        )
    rf_type = response_format.get("type")
    if rf_type not in ("text", "json_object", "json_schema"):
        raise _bad("response_format.type", "Invalid response format type.", rf_type)
    if rf_type != "json_schema":
        return
    schema = response_format.get("json_schema")
    if not isinstance(schema, dict):
        raise _bad(
            "response_format.json_schema", "`json_schema` must be an object.", schema
        )
    name = schema.get("name")
    if not isinstance(name, str) or not name:
        raise _bad(
            "response_format.json_schema.name",
            "`name` must be a non-empty string.",
            name,
        )
    if "schema" not in schema or not isinstance(schema["schema"], dict):
        raise _bad(
            "response_format.json_schema.schema",
            "`schema` must be an object.",
            schema.get("schema"),
        )
    if "strict" in schema and not isinstance(schema["strict"], bool):
        raise _bad(
            "response_format.json_schema.strict",
            "`strict` must be a boolean.",
            schema["strict"],
        )


def normalize_kimi_k3_request(data: Any) -> Any:
    """Validate and normalize a raw chat request when compatibility is enabled.

    This runs before Pydantic field coercion. The input dict is copied so the
    caller's raw request object is not mutated.
    """
    if not kimi_k3_api_compat_enabled() or not isinstance(data, dict):
        return data
    result = dict(data)
    _validate_response_format(result)

    raw_template_kwargs = result.get("chat_template_kwargs")
    if isinstance(raw_template_kwargs, dict):
        template_kwargs: dict[str, Any] = dict(raw_template_kwargs)
    else:
        template_kwargs = {}

    # Keep this bit of request provenance because the normalized ``thinking``
    # object below fills an omitted effort with ``max``. The sampler must still
    # distinguish an implicit default from an explicitly requested effort.
    explicit_effort = isinstance(result.get("thinking"), dict) and (
        "effort" in result["thinking"]
    )
    explicit_effort = explicit_effort or (
        result.get("reasoning_effort") is not None
        and result.get("reasoning_effort") != "none"
    )
    explicit_effort = explicit_effort or "thinking_effort" in template_kwargs
    marker = result.get(KIMI_K3_EFFORT_EXPLICIT_KEY)
    if isinstance(marker, bool):
        explicit_effort = marker
    result[KIMI_K3_EFFORT_EXPLICIT_KEY] = explicit_effort

    # The K3 vendor contract accepts only the string forms of tool_choice.
    # OpenAI's named-function object is parsed by the generic vLLM request
    # model, but K3 has no named-call mode and must reject it at the API edge.
    if isinstance(result.get("tool_choice"), dict):
        raise _bad(
            "tool_choice",
            'Kimi K3 supports only "auto", "none", or "required" for tool_choice.',
            result["tool_choice"],
        )

    if raw_template_kwargs is not None and not isinstance(raw_template_kwargs, dict):
        raise _bad(
            "chat_template_kwargs",
            "`chat_template_kwargs` must be an object.",
            raw_template_kwargs,
        )

    if "thinking" in template_kwargs and not isinstance(
        template_kwargs["thinking"], bool
    ):
        raise _bad(
            "chat_template_kwargs.thinking",
            "`chat_template_kwargs.thinking` must be a boolean.",
            template_kwargs["thinking"],
        )

    for name, default in (
        ("temperature", 0.6),
        ("top_p", 0.95),
    ):
        _validate_number(result, name, default=default)
    for name in ("presence_penalty", "frequency_penalty"):
        _validate_number(result, name, default=0)
    _validate_integer(result, "n", default=1)

    thinking_was_omitted = "thinking" not in result or result.get("thinking") is None
    thinking = result.get("thinking")
    if thinking is None:
        # The OpenAI-compatible/open-source K3 clients use the template
        # keyword instead of the vendor ``thinking`` object.  Keep that
        # fallback when the top-level object is omitted, while preserving the
        # normal precedence of an explicit reasoning_effort.
        template_thinking = template_kwargs.get("thinking")
        if result.get("reasoning_effort") == "none":
            thinking = {"type": "disabled"}
        elif result.get("reasoning_effort") is not None:
            thinking = {"type": "enabled"}
        elif template_thinking is not None:
            thinking = {"type": "enabled" if template_thinking else "disabled"}
        else:
            thinking = {"type": "enabled"}
    elif not isinstance(thinking, dict):
        raise _bad("thinking", "`thinking` must be an object.", thinking)
    unknown = set(thinking) - _THINKING_KEYS
    if unknown:
        raise _bad(
            "thinking", f"Unsupported thinking fields: {sorted(unknown)}", thinking
        )
    thinking = dict(thinking)
    thinking_type = thinking.setdefault("type", "enabled")
    if thinking_was_omitted and result.get("reasoning_effort") == "none":
        thinking_type = thinking["type"] = "disabled"
    if thinking_type not in ("enabled", "disabled"):
        raise _bad(
            "thinking.type", "`type` must be enabled or disabled.", thinking_type
        )
    keep = thinking.get("keep", "all")
    if thinking_type == "enabled" and keep not in _KEEP_VALUES:
        raise _bad("thinking.keep", "`keep` must be all or history.", keep)
    if thinking_type == "disabled":
        # K3 ignores ``keep`` when reasoning is disabled.  Normalize to the
        # canonical value used by the chat template so arbitrary client values
        # cannot cause a template validation failure.
        keep = "all"
    effort = thinking.get("effort")
    if effort is not None and effort not in _EFFORTS:
        raise _bad("thinking.effort", "Invalid thinking effort.", effort)
    if thinking_type == "enabled":
        effort = (
            effort
            or (
                result.get("reasoning_effort")
                if result.get("reasoning_effort") != "none"
                else None
            )
            or template_kwargs.get("thinking_effort")
            or "max"
        )
        if effort not in _EFFORTS:
            raise _bad("reasoning_effort", "Invalid thinking effort.", effort)
        thinking["effort"] = effort
    else:
        # Disabled thinking must stay disabled even when effort is supplied.
        thinking.pop("effort", None)

    kwargs = template_kwargs
    kwargs["thinking"] = thinking_type == "enabled"
    kwargs["keep_reasoning_content"] = keep
    if thinking_type == "enabled":
        kwargs["thinking_effort"] = thinking["effort"]
    else:
        kwargs.pop("thinking_effort", None)
    result["chat_template_kwargs"] = kwargs
    result["thinking"] = thinking
    return result
