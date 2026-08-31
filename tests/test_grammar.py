# ABOUTME: Tests for grammar.py -- drives GenerationState token-by-token toward
# ABOUTME: Uses an oracle to verify known outputs against the vocab and schema.

import json
import string

import pytest

from src.grammar import (
    NO_MATCH_FUNCTION_NAME,
    GenerationState,
    GrammarError,
    Phase,
)
from src.schema import FunctionRegistry
from src.vocab import TokenVocab

REAL_DEFINITIONS_PATH = "data/input/functions_definition.json"


@pytest.fixture(scope="module")
def vocab():
    characters = string.printable
    return TokenVocab(
        raw={char: index for index, char in enumerate(characters)}
    )


@pytest.fixture(scope="module")
def registry():
    return FunctionRegistry.load(REAL_DEFINITIONS_PATH)


def _oracle_pick(state, vocab, target_text_hint):
    """Among valid_token_ids, pick the one whose text is the longest valid
    prefix of the remaining target -- simulates a model that 'wants' to
    produce target_text_hint next, constrained by the mask."""
    valid_ids = state.valid_token_ids(vocab)
    best_id = None
    best_len = -1
    for tid in valid_ids:
        text = vocab.token_text(tid)
        if (
            target_text_hint.startswith(state.progress + text)
            and len(text) > best_len
        ):
            best_id = tid
            best_len = len(text)
    if best_id is None:
        raise AssertionError(
            f"No valid token matches hint {target_text_hint!r}; "
            "valid texts sample: "
            f"{[vocab.token_text(t) for t in list(valid_ids)[:10]]}"
        )
    return best_id


def _run_generation(vocab, registry, prompt, expected, max_steps=500):
    """Drives GenerationState using an oracle that knows the expected output,
    proving the state machine reaches DONE with the correct result when
    guided toward the right answer at every step."""
    state = GenerationState.start(prompt=prompt, registry=registry)
    expected_name = expected["name"]
    expected_params = expected["parameters"]

    steps = 0
    while not state.is_done():
        steps += 1
        if steps > max_steps:
            raise AssertionError(
                f"exceeded {max_steps} steps, stuck in phase {state.phase}"
            )

        phase = state.phase

        if phase == Phase.EXPECT_OPEN_BRACE:
            hint = "{"
        elif phase == Phase.EXPECT_COLON:
            hint = ":"
        elif phase == Phase.EXPECT_KEY:
            target_key = state.candidate_menu[0]
            hint = '"' + target_key + '"'
        elif phase == Phase.EMIT_KNOWN_LITERAL:
            hint = json.dumps(state.prompt)
        elif phase == Phase.EXPECT_ENUM_VALUE:
            hint = '"' + expected_name + '"'
        elif phase == Phase.EXPECT_PARAMETER_ENUM_VALUE:
            val = expected_params[state.current_parameter_name]
            hint = json.dumps(val)
        elif phase == Phase.EXPECT_STRING_OPEN_QUOTE:
            hint = '"'
        elif phase == Phase.EXPECT_COMMA_OR_CLOSE:
            hint = None
        elif phase == Phase.EXPECT_FREE_STRING_VALUE:
            val = expected_params[state.current_parameter_name]
            hint = json.dumps(val, ensure_ascii=False)[1:]
        elif phase == Phase.EXPECT_FREE_NUMBER_VALUE:
            val = expected_params[state.current_parameter_name]
            required = registry.get(expected_name).required_parameter_names()
            is_last = required[-1] == state.current_parameter_name
            hint = str(val) + ("}" if is_last else ",")
        elif phase == Phase.EXPECT_FREE_BOOLEAN_VALUE:
            val = expected_params[state.current_parameter_name]
            hint = "true" if val else "false"
        else:
            hint = None

        if hint is not None:
            token_id = _oracle_pick(state, vocab, hint)
        else:
            valid_ids = state.valid_token_ids(vocab)
            assert len(valid_ids) == 1, (
                f"expected single option at {phase}, got {len(valid_ids)}"
            )
            token_id = next(iter(valid_ids))

        state.advance(token_id, vocab)

    result = state.result()
    assert result == expected, f"MISMATCH:\n  got={result}\n  exp={expected}"
    return result


# --- Test cases covering every function / value type -----------------

@pytest.mark.parametrize(
    "prompt,expected",
    [
        (
            "What is the sum of 2 and 3?",
            {
                "prompt": "What is the sum of 2 and 3?",
                "name": "fn_add_numbers",
                "parameters": {"a": 2, "b": 3},
            },
        ),
        (
            "What is the sum of 265 and 345?",
            {
                "prompt": "What is the sum of 265 and 345?",
                "name": "fn_add_numbers",
                "parameters": {"a": 265, "b": 345},
            },
        ),
        (
            "Greet shrek",
            {
                "prompt": "Greet shrek",
                "name": "fn_greet",
                "parameters": {"name": "shrek"},
            },
        ),
        (
            "Reverse the string 'hello'",
            {
                "prompt": "Reverse the string 'hello'",
                "name": "fn_reverse_string",
                "parameters": {"s": "hello"},
            },
        ),
        (
            "What is the square root of 16?",
            {
                "prompt": "What is the square root of 16?",
                "name": "fn_get_square_root",
                "parameters": {"a": 16},
            },
        ),
        (
            "Calculate the square root of 144",
            {
                "prompt": "Calculate the square root of 144",
                "name": "fn_get_square_root",
                "parameters": {"a": -12.5},
            },
        ),
        (
            "Replace all numbers with NUMBERS",
            {
                "prompt": "Replace all numbers with NUMBERS",
                "name": "fn_substitute_string_with_regex",
                "parameters": {
                    "source_string": "Hello 34 im 233",
                    "regex": "[0-9]+",
                    "replacement": "NUMBERS",
                },
            },
        ),
    ],
)
def test_generation_reaches_expected_result(vocab, registry, prompt, expected):
    _run_generation(vocab, registry, prompt, expected)


def test_registry_can_be_embedded_in_generation_state_without_false_duplicate(
    vocab, registry
):
    """Regression test: FunctionRegistry's model_validator must be
    idempotent, since embedding it as a field of GenerationState causes
    pydantic to re-run validators on the same instance."""
    state = GenerationState.start(prompt="hello", registry=registry)
    assert state.registry.function_names() == registry.function_names()


def test_boolean_parameter_reaches_typed_result(vocab):
    """Generate a boolean without allowing quoted or numeric values."""
    registry = FunctionRegistry(
        functions=[
            {
                "name": "fn_set_enabled",
                "description": "Set an enabled flag.",
                "parameters": {"enabled": {"type": "boolean"}},
                "returns": {"type": "boolean"},
            }
        ]
    )
    expected = {
        "prompt": "Enable the feature",
        "name": "fn_set_enabled",
        "parameters": {"enabled": True},
    }
    _run_generation(vocab, registry, expected["prompt"], expected)


def test_function_without_parameters_reaches_empty_object(vocab):
    """Close the parameters object immediately for a zero-argument call."""
    registry = FunctionRegistry(
        functions=[
            {
                "name": "fn_status",
                "description": "Return status.",
                "parameters": {},
                "returns": {"type": "string"},
            }
        ]
    )
    expected = {
        "prompt": "Show status",
        "name": "fn_status",
        "parameters": {},
    }
    _run_generation(vocab, registry, expected["prompt"], expected)


def test_integer_number_and_enum_parameters_reach_typed_result(vocab):
    """Keep integer values as ints and restrict enum output to its menu."""
    registry = FunctionRegistry(
        functions=[
            {
                "name": "fn_install",
                "description": "Install a firmware release.",
                "parameters": {
                    "version": {"type": "integer"},
                    "firmware": {
                        "type": "string",
                        "enum": ["stable", "beta"],
                    },
                },
                "returns": {"type": "boolean"},
            }
        ]
    )
    expected = {
        "prompt": "Install version 12 using beta firmware",
        "name": "fn_install",
        "parameters": {"version": 12, "firmware": "beta"},
    }
    result = _run_generation(vocab, registry, expected["prompt"], expected)
    assert type(result["parameters"]["version"]) is int


def test_number_parameters_are_collected_as_float(vocab):
    """Match evaluator functions whose number annotations require floats."""
    registry = FunctionRegistry(
        functions=[
            {
                "name": "fn_rate",
                "description": "Set a rate.",
                "parameters": {"rate": {"type": "number"}},
                "returns": {"type": "number"},
            }
        ]
    )
    expected = {
        "prompt": "Set rate to 3",
        "name": "fn_rate",
        "parameters": {"rate": 3.0},
    }
    result = _run_generation(vocab, registry, expected["prompt"], expected)
    assert type(result["parameters"]["rate"]) is float


def test_strings_preserve_quotes_backslashes_and_braces(vocab):
    """Decode valid JSON escapes into exact complex string parameters."""
    registry = FunctionRegistry(
        functions=[
            {
                "name": "fn_format",
                "description": "Format text.",
                "parameters": {"template": {"type": "string"}},
                "returns": {"type": "string"},
            }
        ]
    )
    expected = {
        "prompt": "Format a quoted Windows template",
        "name": "fn_format",
        "parameters": {
            "template": 'Say "hello" to {name} at C:\\Temp',
        },
    }
    _run_generation(vocab, registry, expected["prompt"], expected)


def test_no_matching_function_candidate_raises_clear_error(vocab, registry):
    """Reject an unmatched request instead of inventing a function call."""
    state = GenerationState.start(prompt="Tell me a joke", registry=registry)
    state.phase = Phase.EXPECT_ENUM_VALUE
    state.candidate_menu = [*registry.function_names(), NO_MATCH_FUNCTION_NAME]
    target = f'"{NO_MATCH_FUNCTION_NAME}"'
    with pytest.raises(GrammarError, match="no available function matches"):
        while True:
            token_id = _oracle_pick(state, vocab, target)
            state.advance(token_id, vocab)


def test_no_match_candidate_can_be_disabled(vocab, registry):
    """Exclude the synthetic sentinel for a directly relevant request."""
    state = GenerationState.start(
        prompt="Reverse the string 'hello'",
        registry=registry,
        allow_no_match=False,
    )
    state.phase = Phase.EXPECT_COLON
    state.top_level_key = "name"
    colon_id = vocab.id_of(":")
    assert colon_id is not None
    state.advance(colon_id, vocab)
    assert NO_MATCH_FUNCTION_NAME not in state.candidate_menu
    assert state.candidate_menu == registry.function_names()


def _regex_string_state(registry, progress):
    """Create a state positioned inside the public regex parameter."""
    state = GenerationState.start(
        prompt="Replace all numbers with NUMBER",
        registry=registry,
    )
    state.phase = Phase.EXPECT_FREE_STRING_VALUE
    state.chosen_function_name = "fn_substitute_string_with_regex"
    state.top_level_key = "parameters"
    state.current_parameter_name = "regex"
    state.progress = progress
    return state


def test_quantified_character_class_must_close(vocab, registry):
    """Disallow alternatives after a complete regex such as [0-9]+."""
    state = _regex_string_state(registry, "[0-9]+")
    quote_id = vocab.id_of('"')
    assert quote_id is not None
    assert state.valid_token_ids(vocab) == {quote_id}


def test_bare_character_class_allows_only_close_or_quantifier(vocab, registry):
    """Allow [0-9] to close or become [0-9]+, but never add an alternative."""
    state = _regex_string_state(registry, "[0-9]")
    valid_text = {
        vocab.token_text(token_id)
        for token_id in state.valid_token_ids(vocab)
    }
    assert valid_text == {'"', "+", "*", "?"}


def test_single_token_cannot_cross_completed_regex_boundary(vocab, registry):
    """Reject a token that would append text after a quantified class."""
    extended_vocab = TokenVocab(
        raw={'"': 0, "[0-9]+|": 1, "[0-9]+": 2}
    )
    state = _regex_string_state(registry, "")
    valid_text = {
        extended_vocab.token_text(token_id)
        for token_id in state.valid_token_ids(extended_vocab)
    }
    assert "[0-9]+|" not in valid_text
    assert "[0-9]+" in valid_text


def _prompt_span_state(prompt, progress):
    """Create a free-string state for prompt-boundary tests."""
    registry = FunctionRegistry(
        functions=[
            {
                "name": "fn_extract",
                "description": "Extract text.",
                "parameters": {"value": {"type": "string"}},
                "returns": {"type": "string"},
            }
        ]
    )
    state = GenerationState.start(prompt=prompt, registry=registry)
    state.phase = Phase.EXPECT_FREE_STRING_VALUE
    state.chosen_function_name = "fn_extract"
    state.top_level_key = "parameters"
    state.current_parameter_name = "value"
    state.progress = progress
    return state


def test_exact_quoted_prompt_span_must_close(vocab):
    """Force a quote after an SQL query copied exactly from the request."""
    query = "INSERT INTO logs VALUES (1, 2, 3)"
    state = _prompt_span_state(
        f"Run the query '{query}' on the system database",
        query,
    )
    quote_id = vocab.id_of('"')
    assert quote_id is not None
    assert state.valid_token_ids(vocab) == {quote_id}


def test_token_cannot_overrun_template_suffix():
    """Reject a token that crosses the exact suffix after a colon."""
    vocab = TokenVocab(raw={'"': 0, "!": 1, "! }}": 2})
    state = _prompt_span_state(
        "Format template: Hello {user}'s profile!",
        "Hello {user}'s profile",
    )
    valid_text = {
        vocab.token_text(token_id)
        for token_id in state.valid_token_ids(vocab)
    }
    assert "!" in valid_text
    assert "! }}" not in valid_text


def test_unrequested_leading_space_is_masked_from_file_path():
    """Prevent a path parameter from acquiring a leading blank."""
    vocab = TokenVocab(raw={'"': 0, " /home": 1, "/home": 2})
    state = _prompt_span_state(
        "Read the file at /home/user/data.json with utf-8 encoding",
        "",
    )
    valid_text = {
        vocab.token_text(token_id)
        for token_id in state.valid_token_ids(vocab)
    }
    assert " /home" not in valid_text
    assert "/home" in valid_text

