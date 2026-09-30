# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

from openai.types.responses import FunctionTool
from openai.types.responses.response import ToolChoice as ResponsesToolChoice
from openai.types.responses.tool import Tool as ResponsesTool
from openai.types.responses.tool_choice_allowed import ToolChoiceAllowed
from openai.types.responses.tool_choice_function import ToolChoiceFunction
from xgrammar import StructuralTag, normalize_tool_choice
from xgrammar import get_model_structural_tag as get_xgrammar_model_structural_tag
from xgrammar.openai_tool_call_schema import (
    BuiltinToolParam,
    FunctionToolParam,
)
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    PlusFormat,
    RegexFormat,
    SequenceFormat,
    StarFormat,
    TagFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
)

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.tool_strict_level import ToolStrictLevel

ToolChoice: TypeAlias = (
    Literal["none", "auto", "required"]
    | ChatCompletionNamedToolChoiceParam
    | ResponsesToolChoice
    | None
)
AllowedToolRef: TypeAlias = dict[str, object]
SimplifiedToolChoice: TypeAlias = Literal["auto", "required", "forced"]
StructuralTagBuilder: TypeAlias = Callable[
    [
        list[FunctionToolParam],
        list[BuiltinToolParam],
        SimplifiedToolChoice,
        bool,
        str,
    ],
    StructuralTag,
]

# Keep this list in sync with xgrammar.builtin_structural_tag. It is used for
# vLLM-side validation and for documenting the xgrammar builtin surface that
# can be requested by tool parsers through ``structural_tag_model``.
XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS = frozenset(
    {
        "llama",
        "kimi",
        "deepseek_r1",
        "deepseek_v3_1",
        "qwen_3_5",
        "qwen_3_coder",
        "qwen_3",
        "deepseek_v3_2",
        "glm_4_7",
        "deepseek_v4",
        "deepseek_v4_1",
    }
)
VLLM_BUILTIN_STRUCTURAL_TAG_MODELS = frozenset({"hermes", "hy_v4", "kimi_k3"})
SUPPORTED_STRUCTURAL_TAG_MODELS = (
    XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS | VLLM_BUILTIN_STRUCTURAL_TAG_MODELS
)

_VLLM_STRUCTURAL_TAG_REGISTRY: dict[str, StructuralTagBuilder] = {}


def register_vllm_structural_tag(
    model: str,
) -> Callable[[StructuralTagBuilder], StructuralTagBuilder]:
    """Register a vLLM-owned structural tag builder."""

    def decorator(func: StructuralTagBuilder) -> StructuralTagBuilder:
        _VLLM_STRUCTURAL_TAG_REGISTRY[model] = func
        return func

    return decorator


def _tool_is_strict(tool: ChatCompletionToolsParam | ResponsesTool) -> bool:
    if isinstance(tool, FunctionTool):
        return tool.strict is True
    if isinstance(tool, ChatCompletionToolsParam):
        return tool.function.strict is True
    return False


def _any_tool_strict(
    tools: Sequence[ChatCompletionToolsParam | ResponsesTool],
) -> bool:
    return any(_tool_is_strict(tool) for tool in tools)


def _with_tool_strict(
    tool: ChatCompletionToolsParam | ResponsesTool,
    strict: bool,
) -> ChatCompletionToolsParam | ResponsesTool:
    """Return a copy of ``tool`` with ``strict`` set, leaving the request's alone."""
    if isinstance(tool, FunctionTool):
        return tool.model_copy(update={"strict": strict})
    if isinstance(tool, ChatCompletionToolsParam):
        return tool.model_copy(
            update={"function": tool.function.model_copy(update={"strict": strict})}
        )
    return tool


def _resolve_tool_strictness(
    tools: Sequence[ChatCompletionToolsParam | ResponsesTool],
    tool_choice: ToolChoice,
    strict_level: ToolStrictLevel,
) -> Sequence[ChatCompletionToolsParam | ResponsesTool] | None:
    """Decide whether a structural tag applies and pin each tool's ``strict``.

    A tool without an explicit ``strict`` is treated as non-strict: its call
    envelope is still constrained, but its arguments stay free unless the
    server level is PARAMETER. ``None`` means no structural tag.
    """
    if (
        tool_choice == "auto"
        and strict_level == ToolStrictLevel.AUTO
        and not _any_tool_strict(tools)
    ):
        return None
    return [
        _with_tool_strict(
            tool, strict_level >= ToolStrictLevel.PARAMETER or _tool_is_strict(tool)
        )
        for tool in tools
    ]


def get_model_structural_tag(
    model: str,
    tools: Sequence[ChatCompletionToolsParam | ResponsesTool] | None,
    tool_choice: ToolChoice,
    reasoning: bool,
    token_suffix: str = "",
    strict_level: ToolStrictLevel = ToolStrictLevel.AUTO,
) -> StructuralTag | None:
    """Build a structural tag with xgrammar's builtin model templates."""
    if not tools or tool_choice == "none":
        return None

    tools = _resolve_tool_strictness(tools, tool_choice, strict_level)
    if tools is None:
        return None

    dumped_tools = [_dump_tool_for_xgrammar(tool) for tool in tools]
    dumped_tool_choice = _dump_tool_choice_for_xgrammar(tool_choice)

    if model in _VLLM_STRUCTURAL_TAG_REGISTRY:
        function_tools, builtin_tools, simplified_tool_choice = normalize_tool_choice(
            dumped_tools,
            dumped_tool_choice,
        )
        return _VLLM_STRUCTURAL_TAG_REGISTRY[model](
            function_tools,
            builtin_tools,
            simplified_tool_choice,
            reasoning,
            token_suffix,
        )

    if model not in XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS:
        supported = sorted(SUPPORTED_STRUCTURAL_TAG_MODELS)
        raise ValueError(f"Unknown format type: {model}, supported types: {supported}")

    if token_suffix:
        raise ValueError(
            f"Structural tag model {model!r} is an xgrammar builtin with fixed "
            f"tokens and cannot apply token_suffix={token_suffix!r}"
        )

    return get_xgrammar_model_structural_tag(
        model=model,
        tools=dumped_tools,
        tool_choice=dumped_tool_choice,
        reasoning=reasoning,
    )


def _dump_tool_for_xgrammar(
    tool: ChatCompletionToolsParam | ResponsesTool,
) -> dict[str, Any]:
    """Convert tool objects to xgrammar's Chat Completions tool protocol."""
    if isinstance(tool, FunctionTool):
        function: dict[str, Any] = {"name": tool.name}
        if tool.description is not None:
            function["description"] = tool.description
        if tool.parameters is not None:
            function["parameters"] = tool.parameters
        if tool.strict is not None:
            function["strict"] = tool.strict
        return {"type": "function", "function": function}
    dumped_tool = tool.model_dump(mode="json", exclude_none=True)
    if isinstance(tool, ChatCompletionToolsParam):
        return dumped_tool
    return dict(dumped_tool)


def _dump_tool_choice_for_xgrammar(
    tool_choice: ToolChoice,
) -> dict[str, Any] | str | None:
    """Convert tool_choice objects to xgrammar's expected protocol."""
    if tool_choice is None:
        return None

    if isinstance(tool_choice, str):
        return tool_choice

    if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
        return tool_choice.model_dump(mode="json", exclude_none=True)

    if isinstance(tool_choice, ToolChoiceFunction):
        return {
            "type": "function",
            "function": {"name": tool_choice.name},
        }

    if isinstance(tool_choice, ToolChoiceAllowed):
        return {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": tool_choice.mode,
                "tools": [
                    _dump_allowed_tool_ref_for_xgrammar(tool)
                    for tool in tool_choice.tools
                ],
            },
        }

    return tool_choice.model_dump(mode="json", exclude_none=True)


def _dump_allowed_tool_ref_for_xgrammar(tool_ref: AllowedToolRef) -> AllowedToolRef:
    if (
        tool_ref.get("type") == "function"
        and "function" not in tool_ref
        and "name" in tool_ref
    ):
        return {
            "type": "function",
            "function": {"name": tool_ref["name"]},
        }
    return tool_ref


def get_function_parameters(function) -> dict[str, Any] | bool:
    if getattr(function, "strict", None) is False:
        return True
    return function.parameters if function.parameters is not None else True


def _hermes_tool_tags(tools: list[FunctionToolParam]) -> list[TagFormat]:
    arguments_field_prefix = '", "arguments": '
    formats = [
        # <tool_call>
        # {"name": "t1", "arguments": {"q": "v"}}
        # </tool_call>
        ('<tool_call>\n{"name": "', "}\n</tool_call>"),
        # <tool_call>{"name": "t1", "arguments": {"q": "v"}}</tool_call>
        ('<tool_call>{"name": "', "}</tool_call>"),
    ]

    return [
        TagFormat(
            begin=begin + tool.function.name + arguments_field_prefix,
            content=JSONSchemaFormat(
                json_schema=get_function_parameters(tool.function)
            ),
            end=end,
        )
        for tool in tools
        for begin, end in formats
    ]


@register_vllm_structural_tag("hermes")
def get_hermes_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
    token_suffix: str = "",
) -> StructuralTag:
    del builtin_tools, reasoning, token_suffix

    tool_call_trigger = "<tool_call>"

    if tool_choice == "auto":
        tags = _hermes_tool_tags(tools)
        suffix_tag = (
            TriggeredTagsFormat(triggers=[tool_call_trigger], tags=tags)
            if tags
            else AnyTextFormat()
        )
    elif tool_choice == "forced":
        suffix_tag = TagsWithSeparatorFormat(
            tags=_hermes_tool_tags(tools),
            separator="",
            at_least_one=True,
            stop_after_first=True,
        )
    else:
        suffix_tag = TagsWithSeparatorFormat(
            tags=_hermes_tool_tags(tools),
            separator="",
            at_least_one=True,
        )

    return StructuralTag(format=suffix_tag)


def _minimax_tool_tags(tools: list[FunctionToolParam]) -> list[TagFormat]:
    return [
        TagFormat(
            begin=f'<invoke name="{tool.function.name}">\n',
            content=JSONSchemaFormat(
                json_schema=get_function_parameters(tool.function),
                style="minimax_xml",
            ),
            end="</invoke>\n",
        )
        for tool in tools
    ]


@register_vllm_structural_tag("minimax")
def get_minimax_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
    token_suffix: str = "",
) -> StructuralTag:
    del builtin_tools, reasoning, token_suffix

    tool_call_begin = "<minimax:tool_call>\n"
    tool_call_end = "</minimax:tool_call>"
    tool_call_trigger = "<minimax:tool_call>"

    tags = _minimax_tool_tags(tools)

    if tool_choice == "auto":
        suffix_tag = (
            TriggeredTagsFormat(
                triggers=[tool_call_trigger],
                tags=[
                    TagFormat(
                        begin=tool_call_begin,
                        content=TagsWithSeparatorFormat(
                            tags=tags,
                            separator="",
                            at_least_one=True,
                        ),
                        end=tool_call_end,
                    )
                ],
                excludes=["<think>", "</think>"],
            )
            if tags
            else AnyTextFormat(excludes=["<think>", "</think>"])
        )
    elif tool_choice == "forced":
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value="\n" + tool_call_begin),
                TagsWithSeparatorFormat(
                    tags=tags,
                    separator="",
                    at_least_one=True,
                    stop_after_first=True,
                ),
                ConstStringFormat(value=tool_call_end),
            ]
        )
    else:
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value="\n" + tool_call_begin),
                TagsWithSeparatorFormat(
                    tags=tags,
                    separator="",
                    at_least_one=True,
                ),
                ConstStringFormat(value=tool_call_end),
            ]
        )

    return StructuralTag(format=suffix_tag)


# ---------------------------------------------------------------------------
# Kimi K3 (XTML channel format)
# ---------------------------------------------------------------------------
# K3 assistant output after the reasoning gate (``<|close|>think<|sep|>``):
#   <|open|>response<|sep|> <content> <|close|>response<|sep|>
#   [ <|open|>tools<|sep|>
#       <|open|>call tool="NAME" index="1"<|sep|>
#         <|open|>argument key="K" type="TYPE"<|sep|>VALUE<|close|>argument<|sep|>
#       <|close|>call<|sep|> ...
#     <|close|>tools<|sep|> ]
# See ``encoding_k3.py`` (_render_assistant_segments / _open_tag / _attr) and
# ``kimi_k3_tool_parser.py`` for the exact byte-level encoding this mirrors.
_K3_OPEN = "<|open|>"
_K3_CLOSE = "<|close|>"
_K3_SEP = "<|sep|>"
_K3_RESPONSE_OPEN = f"{_K3_OPEN}response{_K3_SEP}"
_K3_RESPONSE_CLOSE = f"{_K3_CLOSE}response{_K3_SEP}"
_K3_TOOLS_OPEN = f"{_K3_OPEN}tools{_K3_SEP}"
_K3_TOOLS_CLOSE = f"{_K3_CLOSE}tools{_K3_SEP}"
_K3_CALL_CLOSE = f"{_K3_CLOSE}call{_K3_SEP}"
_K3_ARG_CLOSE = f"{_K3_CLOSE}argument{_K3_SEP}"
# The model closes the assistant turn with <|close|>message<|sep|> right before
# the end-of-message token. It is generated (not part of the prompt prefix), so
# the tag must permit it or the FSM would mask the model's natural terminator.
_K3_MESSAGE_CLOSE = f"{_K3_CLOSE}message{_K3_SEP}"
_K3_END_OF_MSG = "<|end_of_msg|>"

# JSON-schema type -> K3 XTML ``type=`` attribute value. Mirrors
# ``encoding_k3._xtml_type`` (integer collapses onto number).
_K3_JSON_TO_XTML_TYPE = {
    "string": "string",
    "integer": "number",
    "number": "number",
    "boolean": "boolean",
    "null": "null",
    "object": "object",
    "array": "array",
}


def _k3_escape_attr(value: str) -> str:
    """Mirror ``encoding_k3._escape_attr_value`` (``&`` then ``"``)."""
    return str(value).replace("&", "&amp;").replace('"', "&quot;")


_K3_STRING_ATOM = r"(?:[^<]|<[^|])"
"""One raw-string character: anything but the ambiguous "<|" marker prefix.
Allows '<' inside values (e.g. HTML snippets); a value *ending* in '<' or
containing a literal "<|" is not expressible and falls back to AnyText via
the pattern checks below never matching those cases at build time (schemas
cannot know values, so the only build-time effect is the length bound)."""


def _k3_bounded_string_regex(prop: dict[str, Any]) -> str | None:
    """Length/pattern constraint for the raw string channel, if expressible.

    The XTML string channel emits values raw (not JSON-quoted), so
    JSONSchemaFormat cannot enforce string constraints there; unconstrained
    AnyText lets maxLength/pattern violations through (observed on the walle
    verifier: over-long junk strings pass the grammar and fail validation).
    xgrammar's regex engine has no lookahead, so the close marker is kept
    unambiguous by excluding the "<|" prefix from value characters.

    Returns a regex for the value, or None to keep permissive AnyText.
    """
    max_len = prop.get("maxLength")
    min_len = prop.get("minLength", 0)
    if not isinstance(max_len, int) or max_len < 0 or max_len > 4096:
        return None
    if not isinstance(min_len, int) or min_len < 0 or min_len > max_len:
        min_len = 0
    return _K3_STRING_ATOM + f"{{{min_len},{max_len}}}"


def _k3_json_pointer_target(document: dict[str, Any], ref: str) -> Any | None:
    """Resolve a local JSON Schema reference using JSON Pointer semantics.

    A property schema is compiled separately from the tool's root schema.  The
    old resolver only handled one definition name, so references such as
    ``#/$defs/group/items`` silently fell back to an unconstrained argument.
    Keep this helper deliberately small and local: external documents are not
    available to the request-time grammar compiler, while local pointers can
    be resolved losslessly (including escaped property names and array indices).
    """
    if ref == "#":
        return document
    if not ref.startswith("#/"):
        return None

    current: Any = document
    for raw_token in ref[2:].split("/"):
        # JSON Pointer decodes ~1 before ~0; doing it in the opposite order
        # would turn an escaped literal ``~1`` into a slash.
        if "~" in raw_token:
            token_chars: list[str] = []
            index = 0
            while index < len(raw_token):
                char = raw_token[index]
                if char != "~":
                    token_chars.append(char)
                    index += 1
                    continue
                if index + 1 >= len(raw_token) or raw_token[index + 1] not in "01":
                    return None
                token_chars.append("/" if raw_token[index + 1] == "1" else "~")
                index += 2
            token = "".join(token_chars)
        else:
            token = raw_token

        if isinstance(current, dict):
            if token not in current:
                return None
            current = current[token]
        elif isinstance(current, list):
            if token == "-" or not token.isdigit():
                return None
            position = int(token)
            if position >= len(current):
                return None
            current = current[position]
        else:
            return None
    return current


def _k3_resolve_root_ref(
    schema: dict[str, Any], root_schema: dict[str, Any] | None
) -> dict[str, Any]:
    """Resolve local ``$ref`` chains while retaining sibling keywords.

    ``root_schema`` is the complete tool-parameter schema, rather than just its
    ``$defs`` map.  This is required for nested pointers and for ``$ref: "#"``
    self references.  Cyclic references are left as the final resolved target;
    JSONSchemaFormat handles the cycle, while ``seen`` prevents this helper from
    looping forever.
    """
    if not root_schema:
        return schema

    resolved = schema
    seen: set[str] = set()
    while True:
        ref = resolved.get("$ref")
        if not isinstance(ref, str) or ref in seen:
            return resolved
        target = _k3_json_pointer_target(root_schema, ref)
        if not isinstance(target, dict):
            return resolved
        seen.add(ref)
        merged = dict(target)
        # JSON Schema allows sibling keywords next to $ref in modern drafts;
        # preserve them when resolving the outer property for XTML typing.
        merged.update({key: value for key, value in resolved.items() if key != "$ref"})
        resolved = merged


def _k3_has_root_ref(value: Any) -> bool:
    """Return whether a schema tree contains a self-reference (``$ref: #``)."""
    if isinstance(value, dict):
        if value.get("$ref") == "#":
            return True
        return any(_k3_has_root_ref(item) for item in value.values())
    if isinstance(value, list):
        return any(_k3_has_root_ref(item) for item in value)
    return False


def _k3_rewrite_root_refs(value: Any, target_ref: str) -> Any:
    """Copy a schema tree, redirecting root self references to ``target_ref``."""
    if isinstance(value, dict):
        return {
            key: target_ref
            if key == "$ref" and item == "#"
            else _k3_rewrite_root_refs(item, target_ref)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_k3_rewrite_root_refs(item, target_ref) for item in value]
    return value


_K3_NO_INSTANCE = object()


def _k3_minimal_instance(
    schema: Any,
    document: dict[str, Any],
    resolving_refs: frozenset[str] = frozenset(),
) -> Any:
    """Build one finite instance of a schema for a recursive edge.

    xgrammar correctly accepts recursive JSON Schema, but a model can keep
    selecting an optional recursive property forever.  When a recursive edge
    is reached we replace that edge with a ``const`` schema.  The value here is
    deliberately a small valid instance of the referenced schema, so the
    resulting grammar remains a subset of the request's original schema.
    """
    if not isinstance(schema, dict):
        return _K3_NO_INSTANCE

    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in resolving_refs:
            return _K3_NO_INSTANCE
        target = _k3_json_pointer_target(document, ref)
        if isinstance(target, dict):
            return _k3_minimal_instance(
                target, document, resolving_refs | frozenset({ref})
            )
        return _K3_NO_INSTANCE

    if "const" in schema:
        return schema["const"]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]

    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list):
            for branch in branches:
                value = _k3_minimal_instance(branch, document, resolving_refs)
                if value is not _K3_NO_INSTANCE:
                    return value
            return _K3_NO_INSTANCE

    all_of = schema.get("allOf")
    if isinstance(all_of, list) and all_of:
        values = [
            _k3_minimal_instance(branch, document, resolving_refs) for branch in all_of
        ]
        if all(value is not _K3_NO_INSTANCE for value in values):
            if all(isinstance(value, dict) for value in values):
                merged: dict[str, Any] = {}
                for value in values:
                    merged.update(value)
                return merged
            if values and all(value == values[0] for value in values[1:]):
                return values[0]
        return _K3_NO_INSTANCE

    json_type = schema.get("type")
    if isinstance(json_type, list):
        for type_name in json_type:
            if isinstance(type_name, str):
                branch = dict(schema)
                branch["type"] = type_name
                value = _k3_minimal_instance(branch, document, resolving_refs)
                if value is not _K3_NO_INSTANCE:
                    return value
        return _K3_NO_INSTANCE

    if json_type == "null":
        return None
    if json_type == "boolean":
        return False
    if json_type == "string":
        minimum = schema.get("minLength", 0)
        maximum = schema.get("maxLength")
        length = minimum if isinstance(minimum, int) and minimum > 0 else 1
        if isinstance(maximum, int) and maximum >= 0:
            length = min(length, maximum)
        return "x" * length
    if json_type in ("integer", "number"):
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)):
            return minimum
        exclusive_minimum = schema.get("exclusiveMinimum")
        if isinstance(exclusive_minimum, (int, float)):
            return exclusive_minimum + 1
        return 0
    if json_type == "array":
        items = schema.get("items")
        minimum_items = schema.get("minItems", 0)
        count = (
            minimum_items if isinstance(minimum_items, int) and minimum_items > 0 else 0
        )
        if isinstance(items, dict):
            item = _k3_minimal_instance(items, document, resolving_refs)
            if count and item is _K3_NO_INSTANCE:
                return _K3_NO_INSTANCE
            return [item for _ in range(count)]
        return []
    if json_type == "object" or isinstance(schema.get("properties"), dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return {}
        required = schema.get("required", [])
        required_names = required if isinstance(required, list) else []
        result: dict[str, Any] = {}
        for name in required_names:
            if not isinstance(name, str) or name not in properties:
                return _K3_NO_INSTANCE
            value = _k3_minimal_instance(properties[name], document, resolving_refs)
            if value is _K3_NO_INSTANCE:
                return _K3_NO_INSTANCE
            result[name] = value
        return result

    return _K3_NO_INSTANCE


def _k3_rewrite_recursive_refs(document: dict[str, Any]) -> dict[str, Any]:
    """Inline local references and replace only the cyclic edges.

    A reference target has to be copied at the reference site for the cycle
    bound to affect the grammar that xgrammar sees.  Keeping a transformed
    target only under ``$defs`` leaves the original reference graph intact and
    still permits unbounded generation.  The expansion is finite because a
    reference already present in ``active_refs`` becomes a valid ``const``
    instance instead of being followed again.
    """

    def expand(value: Any, active_refs: frozenset[str]) -> Any:
        if isinstance(value, list):
            return [expand(item, active_refs) for item in value]
        if not isinstance(value, dict):
            return value

        ref = value.get("$ref")
        if isinstance(ref, str):
            target = _k3_json_pointer_target(document, ref)
            if isinstance(target, dict):
                if ref in active_refs:
                    terminal = _k3_minimal_instance(target, document, active_refs)
                    if terminal is _K3_NO_INSTANCE:
                        # Keep grammar construction total for schemas without
                        # a finite instance; verifier-runnable schemas have a
                        # valid branch and use the more precise value above.
                        terminal = {}
                    return {"const": terminal}

                expanded = expand(target, active_refs | frozenset({ref}))
                if isinstance(expanded, dict):
                    # JSON Schema permits siblings alongside ``$ref``.  Apply
                    # them after expanding the target and keep their own local
                    # references bounded by the same active path.
                    expanded.update(
                        {
                            key: expand(child, active_refs)
                            for key, child in value.items()
                            if key != "$ref"
                        }
                    )
                return expanded

        return {key: expand(child, active_refs) for key, child in value.items()}

    return expand(document, frozenset())


def _k3_schema_alternatives(
    schema: dict[str, Any], root_schema: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Expand root refs and schema unions into scalar-type alternatives."""
    schema = _k3_resolve_root_ref(schema, root_schema)
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list) and branches:
            alternatives: list[dict[str, Any]] = []
            for branch in branches:
                if isinstance(branch, dict):
                    alternatives.extend(_k3_schema_alternatives(branch, root_schema))
            if alternatives:
                return alternatives
            return []

    json_type = schema.get("type")
    if isinstance(json_type, list):
        alternatives = []
        for type_name in json_type:
            if isinstance(type_name, str):
                branch = dict(schema)
                branch["type"] = type_name
                alternatives.append(branch)
        return alternatives
    return [schema] if isinstance(json_type, str) else []


def _k3_argument_tag_for_schema(
    key: str,
    prop: dict[str, Any],
    root_schema: dict[str, Any] | None,
) -> TagFormat | None:
    prop = _k3_resolve_root_ref(prop, root_schema)
    json_type = prop.get("type")
    xtml_type = (
        _K3_JSON_TO_XTML_TYPE.get(json_type) if isinstance(json_type, str) else None
    )
    if xtml_type is None:
        return None
    begin = (
        f'{_K3_OPEN}argument key="{_k3_escape_attr(key)}" type="{xtml_type}"{_K3_SEP}'
    )
    if xtml_type == "string":
        enum_values = prop.get("enum")
        if enum_values is None and isinstance(prop.get("const"), str):
            enum_values = [prop["const"]]
        if (
            isinstance(enum_values, list)
            and enum_values
            and len(enum_values) <= 256
            and all(isinstance(v, str) for v in enum_values)
            and not any("<|" in v for v in enum_values)
        ):
            branches = [ConstStringFormat(value=v) for v in enum_values]
            content: Any = (
                branches[0] if len(branches) == 1 else OrFormat(elements=branches)
            )
        elif (bounded := _k3_bounded_string_regex(prop)) is not None:
            content = RegexFormat(pattern=bounded)
        else:
            content = AnyTextFormat(excludes=[_K3_CLOSE])
    else:
        embedded = prop
        if root_schema:
            embedded = dict(prop)
            # A property is compiled as its own JSON Schema document. Keep the
            # parameter root's definition tables available to local pointers,
            # and redirect ``$ref: "#"`` in recursive properties to a
            # synthetic root definition instead of changing its meaning to the
            # sliced property schema.
            if _k3_has_root_ref(prop):
                root_name = "__k3_root"
                raw_defs = root_schema.get("$defs")
                defs = dict(raw_defs) if isinstance(raw_defs, dict) else {}
                while root_name in defs:
                    root_name = f"_{root_name}"
                root_ref = f"#/$defs/{root_name}"
                root_copy = _k3_rewrite_root_refs(root_schema, root_ref)
                # Use the rewritten copies too: a definition nested under the
                # root may itself contain ``$ref: "#"``.
                raw_defs = root_copy.get("$defs")
                defs = dict(raw_defs) if isinstance(raw_defs, dict) else {}
                raw_definitions = root_copy.get("definitions")
                definitions = (
                    dict(raw_definitions) if isinstance(raw_definitions, dict) else {}
                )
                root_target = {
                    key: value
                    for key, value in root_copy.items()
                    if key not in ("$defs", "definitions", "$id")
                }
                defs[root_name] = root_target
                embedded = _k3_rewrite_root_refs(embedded, root_ref)
                if defs:
                    embedded["$defs"] = defs
                if definitions:
                    embedded["definitions"] = definitions
            else:
                for defs_key in ("$defs", "definitions"):
                    defs_value = root_schema.get(defs_key)
                    if isinstance(defs_value, dict):
                        embedded.setdefault(defs_key, defs_value)
            # xgrammar's recursive JSON-schema support is semantically sound,
            # but a model can repeatedly choose an optional recursive branch
            # and exhaust the generation budget before closing the tool call.
            # Bound only the cyclic edges; ordinary references remain strict.
            embedded = _k3_rewrite_recursive_refs(embedded)
        content = JSONSchemaFormat(json_schema=embedded)
    return TagFormat(begin=begin, content=content, end=_K3_ARG_CLOSE)


def _k3_argument_tag(
    key: str,
    schema: dict[str, Any],
    root_schema: dict[str, Any] | None = None,
) -> Any:
    """Build one ``argument`` XTML tag for property ``key``.

    ``string`` values are emitted raw (bounded by the close marker); every other
    JSON type is emitted as JSON and validated against the property schema.
    Union schemas become alternatives with a fixed XTML ``type=`` attribute.
    Schemas without a concrete type remain permissive so a valid call is never
    rejected.

    ``root_schema`` carries the complete tool parameters document: slicing a
    property out of it orphans ``#/$defs/...`` and ``#`` references, so the
    relevant context is re-attached to keep the embedded schema self-contained.
    """
    if not isinstance(schema, dict):
        return None
    alternatives = _k3_schema_alternatives(schema, root_schema)
    tags = [
        tag
        for alternative in alternatives
        if (tag := _k3_argument_tag_for_schema(key, alternative, root_schema))
        is not None
    ]
    if not tags:
        # Unknown / unconstrained type: constrain the key through the
        # permissive fallback in _k3_arguments_block.
        return None
    return tags[0] if len(tags) == 1 else OrFormat(elements=tags)


def _k3_permissive_argument_tag() -> TagFormat:
    """A key/type-agnostic ``argument`` tag: any attributes, raw value.

    Used as a fallback so tools with union/loose schemas still get the XTML
    skeleton constrained without over-rejecting the value.
    """
    return TagFormat(
        begin=_K3_OPEN + "argument ",
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[^<]*" + _K3_SEP.replace("|", r"\|")),
                AnyTextFormat(excludes=[_K3_CLOSE]),
            ]
        ),
        end=_K3_ARG_CLOSE,
    )


def _k3_arguments_block(parameters: dict[str, Any] | bool) -> Any:
    """Build ``argument`` tags for a tool's parameter schema.

    Require at least one tag when the root schema declares required properties.
    Otherwise, keep accepting zero-or-more tags. Arguments remain order-agnostic
    and non-unique.
    """
    if not isinstance(parameters, dict):
        return StarFormat(content=_k3_permissive_argument_tag())
    props = parameters.get("properties")
    if not isinstance(props, dict) or not props:
        # No declared properties: allow any argument blocks (or none).
        return StarFormat(content=_k3_permissive_argument_tag())
    tags: list[TagFormat] = []
    for key, prop in props.items():
        tag = _k3_argument_tag(key, prop, parameters)
        tags.append(tag if tag is not None else _k3_permissive_argument_tag())
    inner = tags[0] if len(tags) == 1 else OrFormat(elements=list(tags))
    required = parameters.get("required")
    if isinstance(required, list) and required:
        return PlusFormat(content=inner)
    return StarFormat(content=inner)


def _k3_call_tag(tool: FunctionToolParam) -> TagFormat:
    """One ``call`` tag: ``<|open|>call tool="N" index="<digits>"<|sep|> args``."""
    function = tool.function
    parameters = get_function_parameters(function)
    begin = f'{_K3_OPEN}call tool="{_k3_escape_attr(function.name)}" index="'
    return TagFormat(
        begin=begin,
        content=SequenceFormat(
            elements=[
                RegexFormat(pattern=r"[0-9]+"),
                ConstStringFormat(value=f'"{_K3_SEP}'),
                _k3_arguments_block(parameters),
            ]
        ),
        end=_K3_CALL_CLOSE,
    )


def _k3_response_prefix(max_chars: int | None = None) -> list[Any]:
    """The response channel that always precedes the tools channel.

    ``response`` is generated in thinking mode (prefix ends at
    ``<|open|>think<|sep|>``) but is part of the generation prefix in
    non-thinking mode, so its open marker is optional. Excluding the reserved
    marker heads commits ``<|close|>`` to the response-close branch immediately
    and prevents a nested channel from starting inside response text.
    """
    return [
        OptionalFormat(content=ConstStringFormat(value=_K3_RESPONSE_OPEN)),
        TagFormat(
            begin="",
            content=AnyTextFormat(
                excludes=[_K3_OPEN, _K3_CLOSE, _K3_END_OF_MSG],
                max_chars=max_chars,
            ),
            end=_K3_RESPONSE_CLOSE,
        ),
    ]


def _k3_tools_channel(tools: list[FunctionToolParam]) -> TagFormat:
    return TagFormat(
        begin=_K3_TOOLS_OPEN,
        content=TagsWithSeparatorFormat(
            tags=[_k3_call_tag(tool) for tool in tools],
            separator="",
            at_least_one=True,
        ),
        end=_K3_TOOLS_CLOSE,
    )


@register_vllm_structural_tag("kimi_k3")
def get_kimi_k3_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
    token_suffix: str = "",
) -> StructuralTag:
    del builtin_tools, reasoning, token_suffix

    trailer = OptionalFormat(content=ConstStringFormat(value=_K3_MESSAGE_CLOSE))

    # A required tool call must reach the tools channel within the generation
    # budget.  Recursive JSON schemas can otherwise make the model spend the
    # entire budget in free-form response text before it emits the mandatory
    # call.  Keep a short natural-language lead-in while bounding that escape.
    response_max_chars = 256 if tool_choice in ("forced", "required") else None

    if not tools:
        return StructuralTag(
            format=SequenceFormat(
                elements=[*_k3_response_prefix(response_max_chars), trailer]
            )
        )

    if tool_choice == "auto":
        tools_part: Any = OptionalFormat(content=_k3_tools_channel(tools))
    elif tool_choice == "forced":
        # K3 rejects named tool choice upstream; treat defensively as a single
        # mandatory call of the first tool.
        tools_part = _k3_tools_channel(tools[:1])
    else:  # required
        tools_part = _k3_tools_channel(tools)

    return StructuralTag(
        format=SequenceFormat(
            elements=[*_k3_response_prefix(response_max_chars), tools_part, trailer]
        )
    )


# ---------------------------------------------------------------------------
# HYV4 (<tool_calls>/<tool_call>/<arg_key>/<arg_value> structural tokens)
# ---------------------------------------------------------------------------
# HYV4 assistant output after the reasoning gate (``</think:SUF>``):
#   <tool_calls:SUF>
#     <tool_call:SUF>NAME
#       <arg_key:SUF>K</arg_key:SUF><arg_value:SUF>V</arg_value:SUF>
#     </tool_call:SUF>
#   </tool_calls:SUF>
# Adjacent <tool_call> blocks have no separator, and argument values are emitted
# verbatim (no surrounding quotes), so the value body cannot be described by
# JSONSchemaFormat. Only the skeleton (tool names, argument keys, tag order) is
# constrained; each <arg_value> body stays free text.


def _hy_v4_argument_keys(parameters: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Split a tool's argument keys into ``(required, optional)``.

    Both lists keep declaration order. Required keys missing from
    ``properties`` are still treated as required.
    """
    properties = parameters.get("properties") or {}
    required = parameters.get("required") or []
    required_set = set(required)
    required_keys = [k for k in properties if k in required_set]
    required_keys += [k for k in required if k not in properties]
    optional_keys = [k for k in properties if k not in required_set]
    return required_keys, optional_keys


@register_vllm_structural_tag("hy_v4")
def get_hy_v4_structural_tag(
    tools: list[FunctionToolParam],
    builtin_tools: list[BuiltinToolParam],
    tool_choice: SimplifiedToolChoice,
    reasoning: bool,
    token_suffix: str = "",
) -> StructuralTag:
    """Build HYV4 <tool_calls>/<tool_call>/<arg_key>/<arg_value> structural tags.

    Args:
        tools: Normalized function tools the model may call.
        builtin_tools: Unused; HYV4 has no builtin tools.
        tool_choice: Simplified tool choice.
        reasoning: Whether the grammar also covers the reasoning phase.
        token_suffix: Per-checkpoint structural-token suffix including the
            leading colon (e.g. ``":6124c78e"``), or ``""`` when the checkpoint
            uses unsuffixed tokens. The HYV4 tool parser reads it off the
            tokenizer vocab and passes it to ``get_model_structural_tag``.

    """
    del builtin_tools

    think_end = f"</think{token_suffix}>"
    tool_calls_begin = f"<tool_calls{token_suffix}>"
    tool_calls_end = f"</tool_calls{token_suffix}>"
    tool_call_begin = f"<tool_call{token_suffix}>"
    tool_call_end = f"</tool_call{token_suffix}>"
    arg_key_begin = f"<arg_key{token_suffix}>"
    arg_key_end = f"</arg_key{token_suffix}>"
    arg_value_begin = f"<arg_value{token_suffix}>"
    arg_value_end = f"</arg_value{token_suffix}>"

    def arg_pair(key: str) -> SequenceFormat:
        return SequenceFormat(
            elements=[
                ConstStringFormat(value=arg_key_begin),
                ConstStringFormat(value=key),
                ConstStringFormat(value=arg_key_end),
                ConstStringFormat(value=arg_value_begin),
                AnyTextFormat(),
                ConstStringFormat(value=arg_value_end),
            ]
        )

    def single_tool_call(tool: FunctionToolParam) -> TagFormat:
        function = tool.function
        parameters = (
            function.parameters if isinstance(function.parameters, dict) else {}
        )
        required_keys, optional_keys = _hy_v4_argument_keys(parameters)
        elements: list[Any] = [arg_pair(k) for k in required_keys]
        if optional_keys:
            # Optional keys may appear in any order, or not at all.
            optional_pairs = [arg_pair(k) for k in optional_keys]
            elements.append(
                StarFormat(
                    content=optional_pairs[0]
                    if len(optional_pairs) == 1
                    else OrFormat(elements=optional_pairs)
                )
            )
        content: Any = (
            SequenceFormat(elements=elements) if elements else AnyTextFormat()
        )
        return TagFormat(
            begin=tool_call_begin + function.name,
            content=content,
            end=tool_call_end,
        )

    tags = [single_tool_call(tool) for tool in tools]

    if tool_choice == "auto":
        block = TagFormat(
            begin=tool_calls_begin,
            content=TagsWithSeparatorFormat(
                tags=tags,
                separator="",
                at_least_one=True,
            ),
            end=tool_calls_end,
        )
        suffix_tag: Any = TriggeredTagsFormat(triggers=[tool_calls_begin], tags=[block])
    else:
        # HYV4 places tool_call blocks back-to-back with no separator. "forced"
        # has already been filtered down to the single named tool and must emit
        # exactly one block.
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value=tool_calls_begin),
                TagsWithSeparatorFormat(
                    tags=tags,
                    separator="",
                    at_least_one=True,
                    stop_after_first=tool_choice == "forced",
                ),
                ConstStringFormat(value=tool_calls_end),
            ]
        )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    # The reasoning phase is constrained too, so the tag must explicitly skip
    # the ``<think>...</think:SUF>`` prefix.
    prefix_tag = TagFormat(begin="", content=AnyTextFormat(), end=think_end)
    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))
