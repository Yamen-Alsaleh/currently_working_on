"""Unit tests for the generator's greedy token-selection policy."""

from generator import _select_token_id
from grammar import GenerationState, Phase
from schema import FunctionRegistry
from vocab import TokenVocab


def _string_state() -> GenerationState:
    """Create a generation state positioned inside a non-empty string."""
    registry = FunctionRegistry(functions=[])
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
