"""Unit tests for the generator's greedy token-selection policy."""

import json

import pytest

from src.generator import (
    _build_framing_text,
    _normalize_plain_character_set,
    _prompt_overlaps_registry,
    _select_known_token_id,
    _select_token_id,
    generate_function_call,
)
from src.grammar import GenerationState, GrammarError, Phase
from src.schema import FunctionRegistry
from src.vocab import TokenVocab


def _string_state() -> GenerationState:
    """Create a generation state positioned inside a non-empty string."""
    registry = FunctionRegistry(
        functions=[
            {
                "name": "fn_test",
                "description": "test",
                "parameters": {"value": {"type": "string"}},
                "returns": {"type": "string"},
            }
        ]
    )
    state = GenerationState.start(prompt="test", registry=registry)
    state.phase = Phase.EXPECT_FREE_STRING_VALUE
    state.progress = "cat"
    return state


def test_close_quote_wins_when_its_logit_is_close() -> None:
    """Prefer finishing a complete string over a weak prose continuation."""
    vocab = TokenVocab(raw={'"': 0, "more": 1})
    selected = _select_token_id(
        logits=[8.0, 9.0],
        valid_ids={0, 1},
        state=_string_state(),
        vocab=vocab,
    )
    assert selected == 0


def test_strong_continuation_is_not_stopped_early() -> None:
    """Continue a string when the model strongly prefers more content."""
    vocab = TokenVocab(raw={'"': 0, "more": 1})
    selected = _select_token_id(
        logits=[5.0, 9.0],
        valid_ids={0, 1},
        state=_string_state(),
        vocab=vocab,
    )
    assert selected == 1


def test_completed_quoted_value_gets_stronger_close_preference() -> None:
    """Stop an extracted quoted span instead of extending it as a regex."""
    vocab = TokenVocab(raw={'"': 0, ".*cat": 1})
    state = _string_state()
    state.prompt = "Replace 'cat' in 'cat cat'"
    state.current_parameter_name = "regex"
    selected = _select_token_id(
        logits=[1.0, 10.0],
        valid_ids={0, 1},
        state=state,
        vocab=vocab,
    )
    assert selected == 0


def test_completed_character_class_gets_stronger_close_preference() -> None:
    """Stop after a complete character-class regex instead of adding prose."""
    vocab = TokenVocab(raw={'"': 0, "...": 1})
    state = _string_state()
    state.progress = "[aeiouAEIOU]"
    state.current_parameter_name = "regex"
    selected = _select_token_id(
        logits=[1.0, 10.0],
        valid_ids={0, 1},
        state=state,
        vocab=vocab,
    )
    assert selected == 0


def test_single_structural_token_never_calls_the_model() -> None:
    """Return a forced grammar token without requiring logits."""
    vocab = TokenVocab(raw={"{": 0})
    state = GenerationState.start(
        prompt="test",
        registry=FunctionRegistry(
            functions=[
                {
                    "name": "fn_test",
                    "description": "test",
                    "parameters": {},
                    "returns": {"type": "string"},
                }
            ]
        ),
    )
    assert _select_known_token_id({0}, state, vocab) == 0


def test_function_choice_is_not_selected_deterministically() -> None:
    """Keep multiple function-name candidates under LLM control."""
    vocab = TokenVocab(raw={'"': 0, "fn_a": 1, "fn_b": 2})
    state = _string_state()
    state.phase = Phase.EXPECT_ENUM_VALUE
    state.candidate_menu = ["fn_a", "fn_b"]
    assert _select_known_token_id({0, 1, 2}, state, vocab) is None


def _function_selection_state() -> GenerationState:
    """Create a state at the point where a function name is selected."""
    state = _string_state()
    state.phase = Phase.EXPECT_ENUM_VALUE
    state.candidate_menu = ["fn_test", "__no_matching_function__"]
    state.progress = '"'
    return state


def test_weak_no_match_preference_does_not_displace_real_function() -> None:
    """Require a meaningful margin before selecting the error sentinel."""
    vocab = TokenVocab(
        raw={"fn_test": 0, "__no_matching_function__": 1}
    )
    selected = _select_token_id(
        logits=[5.0, 10.0],
        valid_ids={0, 1},
        state=_function_selection_state(),
        vocab=vocab,
    )
    assert selected == 0


def test_strong_no_match_preference_can_still_reject_request() -> None:
    """Keep graceful rejection when the model decisively selects it."""
    vocab = TokenVocab(
        raw={"fn_test": 0, "__no_matching_function__": 1}
    )
    selected = _select_token_id(
        logits=[5.0, 12.0],
        valid_ids={0, 1},
        state=_function_selection_state(),
        vocab=vocab,
    )
    assert selected == 1


def test_plain_character_list_becomes_regex_class() -> None:
    """Repair an unquoted list when only its character class matches."""
    result = {
        "prompt": "Replace all vowels in 'Programming' with asterisks",
        "name": "fn_replace",
        "parameters": {
            "source_string": "Programming",
            "regex": "aeiou",
            "replacement": "*",
        },
    }
    normalized = _normalize_plain_character_set(result["prompt"], result)
    assert normalized["parameters"]["regex"] == "[aeiou]"


def test_literal_regex_remains_unchanged() -> None:
    """Do not alter a plain pattern that already matches literally."""
    result = {
        "prompt": "Replace all cat occurrences with dog",
        "name": "fn_replace",
        "parameters": {
            "source_string": "cat and cat",
            "regex": "cat",
            "replacement": "dog",
        },
    }
    normalized = _normalize_plain_character_set(result["prompt"], result)
    assert normalized["parameters"]["regex"] == "cat"


def test_quoted_absent_pattern_remains_literal() -> None:
    """Respect an explicitly quoted literal even when it has no match."""
    result = {
        "prompt": "Replace all 'abc' in 'a b c' with X",
        "name": "fn_replace",
        "parameters": {
            "source_string": "a b c",
            "regex": "abc",
            "replacement": "X",
        },
    }
    normalized = _normalize_plain_character_set(result["prompt"], result)
    assert normalized["parameters"]["regex"] == "abc"


def test_blank_prompt_is_rejected_before_model_access() -> None:
    """Never force an empty request into the first available function."""
    registry = _string_state().registry
    with pytest.raises(GrammarError, match="prompt must contain"):
        generate_function_call("   ", object(), object(), registry)


def test_framing_lists_enum_choices() -> None:
    """Tell the model about schema-defined enum options."""
    registry = FunctionRegistry(
        functions=[
            {
                "name": "fn_flash",
                "description": "Flash firmware.",
                "parameters": {
                    "firmware": {
                        "type": "string",
                        "enum": ["stable", "beta"],
                    }
                },
                "returns": {"type": "boolean"},
            }
        ]
    )
    framing = _build_framing_text("Flash stable firmware", registry)
    assert 'allowed values: ["stable", "beta"]' in framing


def test_no_match_example_precedes_matching_examples() -> None:
    """Avoid making the rejection example the nearest demonstration."""
    registry = FunctionRegistry.load(
        "data/input/functions_definition.json"
    )
    framing = _build_framing_text(
        "Greet shrek",
        registry,
        allow_no_match=True,
    )
    no_match_position = framing.index(
        "This request does not match any listed function"
    )
    greet_position = framing.index("Greet Ada")
    request_position = framing.rindex("Greet shrek")
    assert no_match_position < greet_position < request_position


def test_direct_schema_overlap_disables_no_match_guidance() -> None:
    """Do not expose the sentinel for an obvious registered operation."""
    registry = FunctionRegistry.load(
        "data/input/functions_definition.json"
    )
    prompt = "Reverse the string 'hello'"
    assert _prompt_overlaps_registry(prompt, registry)
    framing = _build_framing_text(
        prompt,
        registry,
        allow_no_match=False,
    )
    assert "__no_matching_function__" not in framing


def test_every_public_prompt_has_direct_schema_overlap() -> None:
    """Never expose the rejection sentinel in the supplied public batch."""
    registry = FunctionRegistry.load(
        "data/input/functions_definition.json"
    )
    with open(
        "data/input/function_calling_tests.json",
        "r",
        encoding="utf-8",
    ) as stream:
        prompts = [entry["prompt"] for entry in json.load(stream)]
    assert all(
        _prompt_overlaps_registry(prompt, registry)
        for prompt in prompts
    )


def test_unrelated_prompt_keeps_no_match_guidance() -> None:
    """Keep graceful rejection available for unrelated requests."""
    registry = FunctionRegistry.load(
        "data/input/functions_definition.json"
    )
    prompt = "Tell me a joke"
    assert not _prompt_overlaps_registry(prompt, registry)
    framing = _build_framing_text(
        prompt,
        registry,
        allow_no_match=True,
    )
    assert "__no_matching_function__" in framing


def _template_string_state(progress: str) -> GenerationState:
    """Create a template value state with an exact suffix after a colon."""
    registry = FunctionRegistry(
        functions=[
            {
                "name": "fn_format_template",
                "description": "Format a template.",
                "parameters": {"template": {"type": "string"}},
                "returns": {"type": "string"},
            }
        ]
    )
    state = GenerationState.start(
        prompt='Format template: Say "hello" to {name}',
        registry=registry,
    )
    state.phase = Phase.EXPECT_FREE_STRING_VALUE
    state.chosen_function_name = "fn_format_template"
    state.top_level_key = "parameters"
    state.current_parameter_name = "template"
    state.progress = progress
    return state


def test_unique_prompt_span_suffix_is_completed_without_model() -> None:
    """Finish exact prompt text deterministically after its prefix is chosen."""
    vocab = TokenVocab(
        raw={
            " ": 0,
            ' \\"hello\\" to {name}': 1,
            " unrelated": 2,
        }
    )
    state = _template_string_state("Say")
    selected = _select_known_token_id(
        valid_ids={0, 1, 2},
        state=state,
        vocab=vocab,
    )
    assert selected == 1


def test_empty_string_progress_still_requires_model_choice() -> None:
    """Do not choose an extractive string before the LLM begins it."""
    vocab = TokenVocab(raw={"Say": 0, "hello": 1})
    state = _template_string_state("")
    assert _select_known_token_id({0, 1}, state, vocab) is None


def test_unique_prompt_number_suffix_is_completed_without_model() -> None:
    """Finish a long selected numeric literal without repeated inference."""
    registry = FunctionRegistry(
        functions=[
            {
                "name": "fn_interest",
                "description": "Calculate interest.",
                "parameters": {"principal": {"type": "number"}},
                "returns": {"type": "number"},
            }
        ]
    )
    state = GenerationState.start(
        prompt="Calculate interest on 1234567.89 at 0.0375",
        registry=registry,
    )
    state.phase = Phase.EXPECT_FREE_NUMBER_VALUE
    state.chosen_function_name = "fn_interest"
    state.top_level_key = "parameters"
    state.current_parameter_name = "principal"
    state.progress = "1"
    vocab = TokenVocab(raw={"2": 0, "234567.89": 1, "9": 2})
    selected = _select_known_token_id(
        valid_ids={0, 1, 2},
        state=state,
        vocab=vocab,
    )
    assert selected == 1
