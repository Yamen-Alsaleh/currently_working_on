"""Unit tests for the generator's greedy token-selection policy."""

from src.generator import _select_known_token_id, _select_token_id
from src.grammar import GenerationState, Phase
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
