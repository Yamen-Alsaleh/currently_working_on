"""Unit tests for vocabulary loading, indexing, and error handling."""

import json
from pathlib import Path

import pytest

from src.vocab import TokenVocab, VocabError


def write_json(tmp_path: Path, data: object) -> Path:
    """Write JSON test data and return its path."""
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def write_raw(tmp_path: Path, text: str) -> Path:
    """Write raw test text and return its path."""
    path = tmp_path / "vocab.json"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file_raises_vocab_error() -> None:
    """Reject a missing vocabulary file gracefully."""
    with pytest.raises(VocabError):
        TokenVocab.load("data/does_not_exist.json")


def test_malformed_json_raises_vocab_error(tmp_path: Path) -> None:
    """Reject malformed JSON."""
    bad_path = write_raw(tmp_path, '{"a": 0, "b": 1,}')
    with pytest.raises(VocabError):
        TokenVocab.load(bad_path)


def test_non_dict_json_raises_vocab_error(tmp_path: Path) -> None:
    """Require the vocabulary root to be an object."""
    bad_path = write_json(tmp_path, ["a", "b", "c"])
    with pytest.raises(VocabError):
        TokenVocab.load(bad_path)


def test_non_integer_id_raises_vocab_error(tmp_path: Path) -> None:
    """Reject token IDs that are not integers."""
    bad_path = write_json(tmp_path, {"a": "not_an_int"})
    with pytest.raises(VocabError):
        TokenVocab.load(bad_path)


def test_duplicate_ids_raise_vocab_error(tmp_path: Path) -> None:
    """Reject two token strings mapped to the same ID."""
    bad_path = write_json(tmp_path, {"a": 0, "b": 0})
    with pytest.raises(VocabError):
        TokenVocab.load(bad_path)


def test_small_synthetic_vocab_round_trip(tmp_path: Path) -> None:
    """Load and query a portable synthetic Qwen-style vocabulary."""
    good_path = write_json(
        tmp_path,
        {
            "{": 0,
            "}": 1,
            '"': 2,
            ":": 3,
            ",": 4,
            "fn": 5,
            "Ġname": 6,
            "Ġgreet": 7,
        },
    )
    vocab = TokenVocab.load(good_path)

    assert len(vocab) == 8
    name_id = vocab.id_of("Ġname")
    fn_id = vocab.id_of("fn")
    assert name_id is not None
    assert fn_id is not None
    assert vocab.token_text(name_id) == " name"
    assert vocab.token_text(fn_id) == "fn"
    assert vocab.id_of("nope") is None


def test_unknown_id_raises_vocab_error() -> None:
    """Reject an ID absent from the reverse index."""
    vocab = TokenVocab(raw={"a": 0})
    with pytest.raises(VocabError):
        vocab.token_string(99)
