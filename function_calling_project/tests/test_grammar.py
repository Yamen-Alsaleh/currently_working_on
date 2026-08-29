# ABOUTME: Tests for grammar.py -- drives GenerationState token-by-token toward
# ABOUTME: Uses an oracle to verify known outputs against the vocab and schema.

import json
import string

import pytest

from src.grammar import GenerationState, Phase
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
        elif phase == Phase.EXPECT_STRING_OPEN_QUOTE:
            hint = '"'
        elif phase == Phase.EXPECT_COMMA_OR_CLOSE:
            hint = None
        elif phase == Phase.EXPECT_FREE_STRING_VALUE:
            val = expected_params[state.current_parameter_name]
            hint = val + '"'
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
