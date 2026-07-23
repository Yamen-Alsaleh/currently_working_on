# ABOUTME: Tests for vocab.py -- loading and querying the tokenizer vocab file.
# ABOUTME: Covers the real Qwen3-0.6B vocab.json plus synthetic edge cases.

import json

import pytest

from vocab import TokenVocab, VocabError

REAL_VOCAB_PATH = "/goinfre/yalsaleh/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json"


def write_json(tmp_path, data) -> str:
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(data))
    return str(path)


def write_raw(tmp_path, text: str) -> str:
    path = tmp_path / "vocab.json"
    path.write_text(text)
    return str(path)


# --- Real vocab.json (Qwen3-0.6B) ---------------------------------------


def test_loads_real_vocab_file():
    vocab = TokenVocab.load(REAL_VOCAB_PATH)
    assert len(vocab) == 151643


def test_real_vocab_punctuation_tokens_exist_as_single_chars():
    vocab = TokenVocab.load(REAL_VOCAB_PATH)
    assert vocab.id_of("{") is not None
    assert vocab.id_of("}") is not None
    assert vocab.id_of('"') is not None
    assert vocab.id_of(":") is not None
    assert vocab.id_of(",") is not None


def test_real_vocab_token_text_strips_leading_space_marker():
    vocab = TokenVocab.load(REAL_VOCAB_PATH)
    name_id = vocab.id_of("Ġname")
    assert name_id is not None
    assert vocab.token_string(name_id) == "Ġname"
    assert vocab.token_text(name_id) == " name"


def test_real_vocab_token_text_unchanged_without_marker():
    vocab = TokenVocab.load(REAL_VOCAB_PATH)
    fn_id = vocab.id_of("fn")
    assert fn_id is not None
    assert vocab.token_text(fn_id) == "fn"


def test_real_vocab_unknown_string_returns_none():
    vocab = TokenVocab.load(REAL_VOCAB_PATH)
    assert vocab.id_of("this_token_definitely_does_not_exist") is None


def test_real_vocab_unknown_id_raises_vocab_error():
    vocab = TokenVocab.load(REAL_VOCAB_PATH)
    with pytest.raises(VocabError):
        vocab.token_string(999_999_999)


# --- Synthetic edge cases (can't get these from the real file) ---------


def test_missing_file_raises_vocab_error():
    with pytest.raises(VocabError):
        TokenVocab.load("data/does_not_exist.json")


def test_malformed_json_raises_vocab_error(tmp_path):
    bad_path = write_raw(tmp_path, '{"a": 0, "b": 1,}')  # trailing comma

    with pytest.raises(VocabError):
        TokenVocab.load(bad_path)


def test_non_dict_json_raises_vocab_error(tmp_path):
    bad_path = write_json(tmp_path, ["a", "b", "c"])

    with pytest.raises(VocabError):
        TokenVocab.load(bad_path)


def test_non_integer_id_raises_vocab_error(tmp_path):
    bad_path = write_json(tmp_path, {"a": "not_an_int"})

    with pytest.raises(VocabError):
        TokenVocab.load(bad_path)


def test_duplicate_ids_raise_vocab_error(tmp_path):
    # Two different token strings mapping to the same id.
    bad_path = write_json(tmp_path, {"a": 0, "b": 0})

    with pytest.raises(VocabError):
        TokenVocab.load(bad_path)


def test_small_synthetic_vocab_round_trip(tmp_path):
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
    assert vocab.token_text(vocab.id_of("Ġname")) == " name"
    assert vocab.token_text(vocab.id_of("fn")) == "fn"
    assert vocab.id_of("nope") is None
