import json
import re
from typing import Any

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .grammar import GenerationState, GrammarError, Phase
from .schema import FunctionRegistry
from .vocab import TokenVocab

STRING_CLOSE_LOGIT_BIAS = 1.5
COMPLETE_VALUE_CLOSE_LOGIT_BIAS = 12.0


def _build_framing_text(prompt: str, registry: FunctionRegistry) -> str:
    """Build an instruction prompt that makes extraction, not prose, likely."""
    lines = [
        "/no_think",
        "Convert the request to exactly one function-call JSON object.",
        "Copy requested string values exactly; do not explain, continue, "
        "invent text, or write placeholders such as ellipses or 'etc.'.",
        "For regex replacement, source_string is the text being edited, "
        "regex is the matching pattern, and replacement is the new text.",
        "A regex for alternative characters must use a character class: "
        "write [aeiou] for vowels, never the plain text aeiou.",
        "Use the shortest regex that matches only the requested text. "
        "Do not add alternatives or unrelated words after the pattern.",
        "Named replacement symbols mean the literal character: for example, "
        "asterisk or asterisks means *.",
        "",
        "Available functions:",
    ]
    for name in registry.function_names():
        fn = registry.get(name)
        params = ", ".join(
            f"{p} ({fn.parameter_type(p)})"
            for p in fn.required_parameter_names()
        )
        lines.append(f"- {name}({params}): {fn.description}")
    lines.append("")
    lines.append("Examples:")
    if registry.contains("fn_greet"):
        lines.extend(
            [
                "Request: Greet Ada",
                '{"prompt":"Greet Ada","name":"fn_greet",'
                '"parameters":{"name":"Ada"}}',
            ]
        )
    if registry.contains("fn_reverse_string"):
        lines.extend(
            [
                "Request: Reverse the string 'hello'",
                '{"prompt":"Reverse the string \'hello\'",'
                '"name":"fn_reverse_string","parameters":{"s":"hello"}}',
            ]
        )
    if registry.contains("fn_substitute_string_with_regex"):
        lines.extend(
            [
                'Request: Replace all numbers in "Room 12" with NUMBER',
                '{"prompt":"Replace all numbers in \\"Room 12\\" with NUMBER",'
                '"name":"fn_substitute_string_with_regex",'
                '"parameters":{"source_string":"Room 12",'
                '"regex":"[0-9]+","replacement":"NUMBER"}}',
                "Request: Replace vowels in 'Code is fun' with asterisks",
                '{"prompt":"Replace vowels in \'Code is fun\' with asterisks",'
                '"name":"fn_substitute_string_with_regex",'
                '"parameters":{"source_string":"Code is fun",'
                '"regex":"[aeiou]","replacement":"*"}}',
                "Request: Substitute 'red' with 'blue' in 'red car'",
                "{\"prompt\":\"Substitute 'red' with 'blue' in 'red car'\","
                '"name":"fn_substitute_string_with_regex",'
                '"parameters":{"source_string":"red car",'
                '"regex":"red","replacement":"blue"}}',
            ]
        )
    lines.append("")
    lines.append(f"Request: {json.dumps(prompt)}")
    lines.append("JSON:")
    return "\n".join(lines)


def _select_token_id(
    logits: list[float],
    valid_ids: set[int],
    state: GenerationState,
    vocab: TokenVocab,
) -> int:
    """
    Select the best valid token, with a small completed-string preference.
    """
    best_id = max(valid_ids, key=lambda token_id: logits[token_id])
    if state.phase != Phase.EXPECT_FREE_STRING_VALUE or not state.progress:
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
    return None


def _looks_like_complete_value(state: GenerationState) -> bool:
    """Recognize common boundaries without choosing a parameter's content."""
    value = state.progress
    quoted_values = {
        match.group(2)
        for match in re.finditer(r"(['\"])(.*?)\1", state.prompt)
    }
    if value in quoted_values:
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
        match.group(2) for match in re.finditer(r"(['\"])(.*?)\1", prompt)
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
    framing_text = _build_framing_text(prompt, registry)
    tensor = model.encode(framing_text)
    input_ids = tensor[0].tolist()
    state = GenerationState.start(prompt=prompt, registry=registry)
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
    result = state.result()
    return _normalize_plain_character_set(prompt, result)
