# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.envs as envs
from vllm.entrypoints.openai.chat_completion.kimi_k3_compat import (
    normalize_kimi_k3_request,
)
from vllm.entrypoints.openai.chat_completion.kimi_k3_tools import (
    effective_tool_objects,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.exceptions import VLLMValidationError


def _payload(**extra):
    return {
        "model": "kimi-k3",
        "messages": [{"role": "user", "content": "hello"}],
        **extra,
    }


def test_k3_compat_disabled_preserves_request_and_template_path(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_KIMI_K3_API_COMPAT", False)
    payload = _payload(thinking={"type": "enabled", "effort": "bogus"})
    request = ChatCompletionRequest.model_validate(payload)

    assert request.thinking == payload["thinking"]
    params = request.build_chat_params(None, "string")
    assert params.chat_template_kwargs.get("thinking") is None

    # Unknown vendor values remain accepted when the opt-in contract is off.
    assert ChatCompletionRequest.model_validate(_payload(thinking="legacy"))


def test_k3_compat_defaults_and_precedence(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_KIMI_K3_API_COMPAT", True)
    request = ChatCompletionRequest.model_validate(
        _payload(
            thinking={"type": "enabled", "effort": "low", "keep": "history"},
            reasoning_effort="max",
        )
    )

    assert request.thinking == {
        "type": "enabled",
        "effort": "low",
        "keep": "history",
    }
    kwargs = request.chat_template_kwargs
    assert kwargs is not None
    assert kwargs["thinking"] is True
    assert kwargs["thinking_effort"] == "low"
    assert kwargs["keep_reasoning_content"] == "history"


def test_k3_compat_reasoning_effort_and_disabled(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_KIMI_K3_API_COMPAT", True)
    request = ChatCompletionRequest.model_validate(_payload(reasoning_effort="high"))
    assert request.thinking == {"type": "enabled", "effort": "high"}
    disabled = ChatCompletionRequest.model_validate(_payload(reasoning_effort="none"))
    assert disabled.thinking == {"type": "disabled"}
    assert disabled.chat_template_kwargs is not None
    assert disabled.chat_template_kwargs["thinking"] is False


@pytest.mark.parametrize(
    ("template_thinking", "expected_type"),
    [(False, "disabled"), (True, "enabled")],
)
def test_k3_compat_uses_opensource_template_thinking_when_top_level_omitted(
    monkeypatch, template_thinking, expected_type
):
    monkeypatch.setattr(envs, "VLLM_KIMI_K3_API_COMPAT", True)
    request = ChatCompletionRequest.model_validate(
        _payload(chat_template_kwargs={"thinking": template_thinking})
    )

    assert request.thinking["type"] == expected_type
    assert request.chat_template_kwargs is not None
    assert request.chat_template_kwargs["thinking"] is template_thinking


def test_k3_compat_reasoning_effort_overrides_opensource_template_thinking(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_KIMI_K3_API_COMPAT", True)
    request = ChatCompletionRequest.model_validate(
        _payload(
            reasoning_effort="low",
            chat_template_kwargs={"thinking": False},
        )
    )

    assert request.thinking == {"type": "enabled", "effort": "low"}
    assert request.chat_template_kwargs is not None
    assert request.chat_template_kwargs["thinking"] is True


@pytest.mark.parametrize(
    "chat_template_kwargs",
    ["oops", ["thinking", False], {"thinking": "false"}],
)
def test_k3_compat_rejects_malformed_template_kwargs(monkeypatch, chat_template_kwargs):
    monkeypatch.setattr(envs, "VLLM_KIMI_K3_API_COMPAT", True)
    with pytest.raises(VLLMValidationError):
        normalize_kimi_k3_request(_payload(chat_template_kwargs=chat_template_kwargs))


@pytest.mark.parametrize(
    "extra",
    [
        {"thinking": "enabled"},
        {"thinking": {"type": "enabled", "effort": "invalid"}},
        {"thinking": {"type": "enabled", "keep": "none"}},
        {"top_p": 0.8},
        {"presence_penalty": 0.5},
        {"n": 2},
        {"response_format": {"type": "bogus"}},
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "x", "schema": {}, "strict": "yes"},
            }
        },
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": {}},
            }
        },
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "x"},
            }
        },
    ],
)
def test_k3_compat_rejects_raw_invalid_values(monkeypatch, extra):
    monkeypatch.setattr(envs, "VLLM_KIMI_K3_API_COMPAT", True)
    with pytest.raises(VLLMValidationError):
        normalize_kimi_k3_request(_payload(**extra))


def _weather_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_k3_dynamic_system_tools_are_preserved_and_effective(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_KIMI_K3_API_COMPAT", True)
    request = ChatCompletionRequest.model_validate(
        _payload(
            messages=[
                {"role": "system", "content": "", "tools": [_weather_tool()]},
                {"role": "user", "content": "weather?"},
            ],
            tool_choice="required",
        )
    )

    assert request.tool_choice == "required"
    assert request.messages[0]["tools"]
    tools = effective_tool_objects(request)
    assert [tool.function.name for tool in tools] == ["get_weather"]
    assert request.build_chat_params(None, "string").tool_choice == "required"


def test_k3_dynamic_tool_name_must_not_start_with_digit(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_KIMI_K3_API_COMPAT", True)
    tool = _weather_tool()
    tool["function"]["name"] = "1bad_name"
    with pytest.raises(VLLMValidationError):
        ChatCompletionRequest.model_validate(
            _payload(
                messages=[
                    {"role": "system", "content": "", "tools": [tool]},
                    {"role": "user", "content": "weather?"},
                ]
            )
        )


def test_k3_dynamic_tool_orphan_result_is_rejected(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_KIMI_K3_API_COMPAT", True)
    with pytest.raises(VLLMValidationError, match="tool_call_id"):
        ChatCompletionRequest.model_validate(
            _payload(
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "tool", "tool_call_id": "missing", "content": "x"},
                ]
            )
        )


def test_k3_parser_receives_dynamic_tools_for_structural_tag(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_KIMI_K3_API_COMPAT", True)
    request = ChatCompletionRequest.model_validate(
        _payload(
            messages=[
                {"role": "system", "content": "", "tools": [_weather_tool()]},
                {"role": "user", "content": "weather?"},
            ],
            tool_choice="required",
        )
    )

    from vllm.parser.kimi_k3 import KimiK3Parser
    from vllm.tool_parsers.kimi_k3_tool_parser import KimiK3ToolParser

    class _Tokenizer:
        def get_vocab(self):
            return {}

    class _Parser(KimiK3Parser):
        reasoning_parser_cls = None
        tool_parser_cls = KimiK3ToolParser

    parser = _Parser(_Tokenizer(), effective_tool_objects(request))
    adjusted = parser.adjust_request(request)
    assert adjusted.tools is None
    assert [tool.function.name for tool in effective_tool_objects(request)] == [
        "get_weather"
    ]
    assert adjusted.structured_outputs is not None
    assert adjusted.structured_outputs.structural_tag is not None


def test_k3_required_tool_call_gets_default_thinking_budget(monkeypatch):
    """Implicit K3 reasoning leaves room for a required XTML tool call."""
    monkeypatch.setattr(envs, "VLLM_KIMI_K3_API_COMPAT", True)
    request = ChatCompletionRequest.model_validate(
        _payload(
            tools=[_weather_tool()],
            tool_choice="required",
            thinking={"type": "enabled"},
        )
    )

    params = request.to_sampling_params(2048, {})

    assert params.thinking_token_budget == 256


def test_k3_required_tool_call_preserves_explicit_budget_and_effort(monkeypatch):
    """Explicit request controls must not be replaced by the default cap."""
    monkeypatch.setattr(envs, "VLLM_KIMI_K3_API_COMPAT", True)
    explicit_budget = ChatCompletionRequest.model_validate(
        _payload(
            tools=[_weather_tool()],
            tool_choice="required",
            thinking={"type": "enabled"},
            thinking_token_budget=512,
        )
    )
    explicit_effort = ChatCompletionRequest.model_validate(
        _payload(
            tools=[_weather_tool()],
            tool_choice="required",
            thinking={"type": "enabled", "effort": "high"},
        )
    )
    explicit_unlimited = ChatCompletionRequest.model_validate(
        _payload(
            tools=[_weather_tool()],
            tool_choice="required",
            thinking={"type": "enabled"},
            thinking_token_budget=-1,
        )
    )

    assert explicit_budget.to_sampling_params(2048, {}).thinking_token_budget == 512
    assert explicit_effort.to_sampling_params(2048, {}).thinking_token_budget is None
    assert explicit_unlimited.to_sampling_params(2048, {}).thinking_token_budget is None


def test_k3_default_thinking_budget_only_applies_to_required_tools(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_KIMI_K3_API_COMPAT", True)
    no_tools = ChatCompletionRequest.model_validate(
        _payload(thinking={"type": "enabled"})
    )
    optional_tool = ChatCompletionRequest.model_validate(
        _payload(tools=[_weather_tool()], tool_choice="auto")
    )
    disabled = ChatCompletionRequest.model_validate(
        _payload(
            tools=[_weather_tool()],
            tool_choice="required",
            thinking={"type": "disabled"},
        )
    )

    assert no_tools.to_sampling_params(2048, {}).thinking_token_budget is None
    assert optional_tool.to_sampling_params(2048, {}).thinking_token_budget is None
    assert disabled.to_sampling_params(2048, {}).thinking_token_budget is None
