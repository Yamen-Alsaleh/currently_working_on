import json
import re
from typing import Any

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .grammar import (
    NO_MATCH_FUNCTION_NAME,
    GenerationState,
    GrammarError,
    Phase,
)
from .schema import FunctionRegistry, SchemaError
from .vocab import TokenVocab

STRING_CLOSE_LOGIT_BIAS = 1.5
COMPLETE_VALUE_CLOSE_LOGIT_BIAS = 12.0
NO_MATCH_LOGIT_PENALTY = 6.0


def _append_example(
    lines: list[str],
    request: str,
    name: str,
    parameters: dict[str, Any],
) -> None:
    """Append one consistently escaped function-call example."""
    example = {
        "prompt": request,
        "name": name,
        "parameters": parameters,
    }
    lines.extend(
        [
            f"Request: {json.dumps(request, ensure_ascii=False)}",
            json.dumps(example, ensure_ascii=False, separators=(",", ":")),
        ]
    )


def _build_framing_text(
    prompt: str,
    registry: FunctionRegistry,
    allow_no_match: bool = True,
) -> str:
    """Build an instruction prompt that makes extraction, not prose, likely."""
    lines = [
        "/no_think",
        "Output exactly one function-call JSON object for the request.",
        "Copy string values exactly and stop at their request boundary. "
        "Do not explain, continue, or invent text.",
        "Escape backslashes and double quotes inside JSON string values.",
    ]
    if registry.contains("fn_substitute_string_with_regex"):
        lines.extend(
            [
                "For regex replacement, source_string is the edited text, "
                "regex is the shortest matching pattern, and replacement "
                "is the new text.",
                "Use [aeiou] for alternative characters. Named symbols are "
                "literal: asterisk means *.",
            ]
        )
    if allow_no_match:
        lines.append(
            f"If no listed function can satisfy the request, select "
            f"{NO_MATCH_FUNCTION_NAME}."
        )
    lines.extend(["", "Available functions:"])
    for name in registry.function_names():
        fn = registry.get(name)
        formatted_params = []
        for parameter_name in fn.required_parameter_names():
            parameter = fn.parameter_schema(parameter_name)
            details = parameter.type
            if parameter.enum is not None:
                choices = json.dumps(parameter.enum, ensure_ascii=False)
                details += f", allowed values: {choices}"
            formatted_params.append(f"{parameter_name} ({details})")
        params = ", ".join(formatted_params)
        lines.append(f"- {name}({params}): {fn.description}")
    lines.append("")
    lines.append("Examples:")
    if allow_no_match:
        _append_example(
            lines,
            "This request does not match any listed function",
            NO_MATCH_FUNCTION_NAME,
            {},
        )
    if registry.contains("fn_greet"):
        _append_example(
            lines,
            "Greet Ada",
            "fn_greet",
            {"name": "Ada"},
        )
    if registry.contains("fn_reverse_string"):
        _append_example(
            lines,
            "Reverse the string 'hello'",
            "fn_reverse_string",
            {"s": "hello"},
        )
    if registry.contains("fn_substitute_string_with_regex"):
        _append_example(
            lines,
            'Replace all numbers in "Room 12" with NUMBER',
            "fn_substitute_string_with_regex",
            {
                "source_string": "Room 12",
                "regex": "[0-9]+",
                "replacement": "NUMBER",
            },
        )
        _append_example(
            lines,
            "Replace vowels in 'Code is fun' with asterisks",
            "fn_substitute_string_with_regex",
            {
                "source_string": "Code is fun",
                "regex": "[aeiou]",
                "replacement": "*",
            },
        )
        _append_example(
            lines,
            "Substitute 'red' with 'blue' in 'red car'",
            "fn_substitute_string_with_regex",
            {
                "source_string": "red car",
                "regex": "red",
                "replacement": "blue",
            },
        )
    if registry.contains("fn_execute_sql_query"):
        _append_example(
            lines,
            "Execute SQL query 'SELECT id FROM users' on staging database",
            "fn_execute_sql_query",
            {"query": "SELECT id FROM users", "database": "staging"},
        )
    if registry.contains("fn_read_file"):
        _append_example(
            lines,
            r"Read C:\Temp\settings.ini with utf-8 encoding",
            "fn_read_file",
            {"path": r"C:\Temp\settings.ini", "encoding": "utf-8"},
        )
    if registry.contains("fn_format_template"):
        _append_example(
            lines,
            'Format template: Say "welcome" to {user}',
            "fn_format_template",
            {"template": 'Say "welcome" to {user}'},
        )
    lines.append("")
    lines.append(f"Request: {json.dumps(prompt)}")
    lines.append("JSON:")
    return "\n".join(lines)


def _normalized_terms(text: str) -> set[str]:
    """Return lowercase word stems used for conservative relevance checks."""
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "please",
        "that",
        "the",
        "this",
        "to",
        "what",
        "with",
    }
    terms: set[str] = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) < 2 or word in stop_words:
            continue
        terms.add(word)
        if word.endswith("ing") and len(word) > 5:
            base = word[:-3]
            terms.add(base)
            if len(base) > 2 and base[-1] == base[-2]:
                terms.add(base[:-1])
        elif word.endswith("ed") and len(word) > 4:
            base = word[:-2]
            terms.add(base)
            if len(base) > 2 and base[-1] == base[-2]:
                terms.add(base[:-1])
        elif word.endswith("s") and len(word) > 3:
            terms.add(word[:-1])
    return terms


def _prompt_overlaps_registry(
    prompt: str,
    registry: FunctionRegistry,
) -> bool:
    """Return whether the request shares a meaningful term with the schema.

    This check never selects a function. It only decides whether the synthetic
    no-match sentinel is necessary. Actual function selection remains an LLM
    logit choice constrained to registered names.
    """
    prompt_terms = _normalized_terms(prompt)
    schema_terms: set[str] = set()
    for function_name in registry.function_names():
        function = registry.get(function_name)
        schema_terms.update(_normalized_terms(function_name))
        schema_terms.update(_normalized_terms(function.description))
        for parameter_name in function.required_parameter_names():
            schema_terms.update(_normalized_terms(parameter_name))
    return bool(prompt_terms & schema_terms)


def _select_token_id(
    logits: list[float],
    valid_ids: set[int],
    state: GenerationState,
    vocab: TokenVocab,
) -> int:
    """
    Select the best valid token, with a small completed-string preference.
    """
    no_match_only_ids = _no_match_only_token_ids(valid_ids, state, vocab)

    def adjusted_logit(token_id: int) -> float:
        """Return a score that requires confidence before rejecting input."""
        penalty = (
            NO_MATCH_LOGIT_PENALTY
            if token_id in no_match_only_ids
            else 0.0
        )
        return logits[token_id] - penalty

    best_id = max(valid_ids, key=adjusted_logit)
    if (
        state.phase != Phase.EXPECT_FREE_STRING_VALUE
        or not state.progress
        or state.string_escape_pending
    ):
        return best_id

    quote_id = vocab.id_of('"')
    close_bias = STRING_CLOSE_LOGIT_BIAS
    if _looks_like_complete_value(state):
        close_bias = COMPLETE_VALUE_CLOSE_LOGIT_BIAS
    if (
        quote_id is not None
        and quote_id in valid_ids
        and logits[quote_id] + close_bias >= logits[best_id]
    ):
        return quote_id
    return best_id


def _no_match_only_token_ids(
    valid_ids: set[int],
    state: GenerationState,
    vocab: TokenVocab,
) -> set[int]:
    """Find tokens that commit function selection to the no-match sentinel.

    The sentinel remains available so an unrelated non-blank request can fail
    gracefully. A small penalty applies only at the point where a token rules
    out every real function. This prevents a synthetic error marker from
    beating an ordinary function on a weak preference while preserving the
    model's ability to reject a request when its preference is decisive.
    """
    if (
        state.phase != Phase.EXPECT_ENUM_VALUE
        or NO_MATCH_FUNCTION_NAME not in state.candidate_menu
    ):
        return set()

    sentinel = f'"{NO_MATCH_FUNCTION_NAME}"'
    real_targets = [
        f'"{candidate}"'
        for candidate in state.candidate_menu
        if candidate != NO_MATCH_FUNCTION_NAME
    ]
    result = {
        token_id
        for token_id in valid_ids
        if sentinel.startswith(state.progress + vocab.token_text(token_id))
        and not any(
            target.startswith(state.progress + vocab.token_text(token_id))
            for target in real_targets
        )
    }
    if result == valid_ids:
        return set()
    return result


def _select_known_token_id(
    valid_ids: set[int],
    state: GenerationState,
    vocab: TokenVocab,
) -> int | None:
    """Select deterministic grammar output without an unnecessary LLM call.

    Structural punctuation and already-known literals contain no semantic
    decision. Choosing their longest valid token preserves the exact text
    while reducing repeated full-context inference on CPU. Function names
    and free parameter values remain model-selected.
    """
    if len(valid_ids) == 1:
        return next(iter(valid_ids))
    known_literal = state.phase == Phase.EMIT_KNOWN_LITERAL
    known_key = (
        state.phase == Phase.EXPECT_KEY and len(state.candidate_menu) == 1
    )
    if known_literal or known_key:
        return max(
            valid_ids,
            key=lambda token_id: len(vocab.token_text(token_id)),
        )
    boundary_suffix = state.unique_prompt_boundary_suffix()
    if boundary_suffix:
        boundary_ids = {
            token_id
            for token_id in valid_ids
            if boundary_suffix.startswith(vocab.token_text(token_id))
        }
        if boundary_ids:
            return max(
                boundary_ids,
                key=lambda token_id: len(vocab.token_text(token_id)),
            )
    number_suffix = state.unique_prompt_number_suffix()
    if number_suffix:
        number_ids = {
            token_id
            for token_id in valid_ids
            if number_suffix.startswith(vocab.token_text(token_id))
        }
        if number_ids:
            return max(
                number_ids,
                key=lambda token_id: len(vocab.token_text(token_id)),
            )
    return None


def _looks_like_complete_value(state: GenerationState) -> bool:
    """Recognize common boundaries without choosing a parameter's content."""
    value = state.decoded_string_progress()
    if value is None:
        return False
    quoted_values = {
        match.group(2)
        for match in re.finditer(r"(['\"])(.*?)\1", state.prompt)
    }
    if value in quoted_values:
        return True
    if ":" in state.prompt and value == state.prompt.split(":", 1)[1].strip():
        return True

    parameter_name = (state.current_parameter_name or "").lower()
    is_pattern = "regex" in parameter_name or "pattern" in parameter_name
    if is_pattern and re.search(r"\](?:[+*?])?$", value):
        return True

    is_replacement = (
        "replacement" in parameter_name
        or "substitution" in parameter_name
        or "replace_with" in parameter_name
    )
    return is_replacement and len(value) == 1 and not value.isalnum()


def _normalize_plain_character_set(
    prompt: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Repair a model-emitted character list into a usable regex class.

    A small model can identify a character group while omitting regex
    brackets. This safeguard applies only when the request asks for all or
    every occurrence, the pattern is an unquoted alphabetic list, the literal
    list does not match the source, and its character-class form does match.
    Literal word, phrase, substring, and quoted patterns remain unchanged.
    """
    parameters = result.get("parameters")
    if not isinstance(parameters, dict):
        return result

    pattern_name = next(
        (
            name
            for name in parameters
            if "regex" in name.lower() or "pattern" in name.lower()
        ),
        None,
    )
    source_name = next(
        (
            name
            for name in ("source_string", "source", "text")
            if isinstance(parameters.get(name), str)
        ),
        None,
    )
    if pattern_name is None or source_name is None:
        return result

    pattern = parameters.get(pattern_name)
    source = parameters.get(source_name)
    if (
        not isinstance(pattern, str)
        or not isinstance(source, str)
        or len(pattern) < 2
        or not pattern.isalpha()
    ):
        return result

    lowered_prompt = prompt.lower()
    asks_for_all = re.search(r"\b(?:all|every)\b", lowered_prompt)
    asks_for_literal = re.search(
        r"\b(?:word|phrase|substring)\b",
        lowered_prompt,
    )
    quoted_values = {
        match.group(2)
        for match in re.finditer(r"(['\"])(.*?)\1", prompt)
    }
    if (
        asks_for_all is None
        or asks_for_literal is not None
        or pattern in quoted_values
    ):
        return result

    try:
        if re.search(pattern, source) is not None:
            return result
        character_class = f"[{re.escape(pattern)}]"
        if re.search(character_class, source) is None:
            return result
    except re.error:
        return result

    normalized = dict(result)
    normalized_parameters = dict(parameters)
    normalized_parameters[pattern_name] = character_class
    normalized["parameters"] = normalized_parameters
    return normalized


def generate_function_call(
    prompt: str,
    model: Small_LLM_Model,
    vocab: TokenVocab,
    registry: FunctionRegistry,
    max_steps: int = 500,
) -> dict[str, Any]:
    """Generate one schema-conforming function call with greedy decoding."""
    if not prompt.strip():
        raise GrammarError(
            "prompt must contain at least one non-space character"
        )
    allow_no_match = not _prompt_overlaps_registry(prompt, registry)
    framing_text = _build_framing_text(
        prompt,
        registry,
        allow_no_match,
    )
    tensor = model.encode(framing_text)
    input_ids = tensor[0].tolist()
    state = GenerationState.start(
        prompt=prompt,
        registry=registry,
        allow_no_match=allow_no_match,
    )
    current_step = 0
    while not state.is_done() and current_step < max_steps:
        valid_ids = state.valid_token_ids(vocab)
        if not valid_ids:
            raise GrammarError(
                f"no valid tokens in generation phase {state.phase.name}"
            )
        token_id = _select_known_token_id(valid_ids, state, vocab)
        if token_id is None:
            logits = model.get_logits_from_input_ids(input_ids)
            token_id = _select_token_id(logits, valid_ids, state, vocab)
        input_ids.append(token_id)
        state.advance(token_id, vocab)
        current_step += 1
    if not state.is_done():
        raise GrammarError(
            "couldn't generate function call for the prompt "
            f'"{prompt}" within max_steps={max_steps}'
        )
    result = _normalize_plain_character_set(prompt, state.result())
    try:
        return registry.validate_call(result, expected_prompt=prompt)
    except SchemaError as error:
        message = f"generated call failed schema validation: {error}"
        raise GrammarError(message) from error
