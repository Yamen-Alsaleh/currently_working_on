"""Validated token vocabulary helpers for constrained decoding."""

import json
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, PrivateAttr, ValidationError, model_validator


class VocabError(Exception):
    """Report an invalid vocabulary or an unknown token ID."""


class TokenVocab(BaseModel):
    """Index a tokenizer vocabulary by ID and decoded first character."""

    raw: dict[str, int]
    _id_to_token: dict[int, str] = PrivateAttr(default_factory=dict)
    _by_first_char: dict[str, list[int]] = PrivateAttr(default_factory=dict)
    _string_value_ids_cache: dict[bool, set[int]] = PrivateAttr(
        default_factory=dict
    )

    @model_validator(mode="after")
    def build(self) -> "TokenVocab":
        """Build reverse and first-character indexes after validation."""
        self._id_to_token = {}
        for token, token_id in self.raw.items():
            if token_id in self._id_to_token:
                raise ValueError(f"duplicate token: {token!r}")
            else:
                self._id_to_token[token_id] = token

        self._by_first_char = {}
        for token_id in self._id_to_token:
            text = self.token_text(token_id)
            if text == "":
                continue
            first = text[0]
            self._by_first_char.setdefault(first, []).append(token_id)
        return self

    @classmethod
    def load(cls, path: str | Path) -> "TokenVocab":
        """Load a tokenizer vocab JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as e:
                    raise VocabError(f"{path} is not valid JSON: {e}") from e

                if not isinstance(parsed, dict):
                    raise VocabError(
                        f"{path} must contain a JSON object,"
                        + f" got {type(parsed).__name__}"
                    )
                try:
                    return cls(raw=parsed)
                except ValidationError as e:
                    raise VocabError(
                        f"{path} does not match the expected schema: {e}"
                    ) from e

        except (OSError, UnicodeDecodeError) as e:
            raise VocabError(f"could not read {path}: {e}") from e

    def ids_starting_with(self, char: str) -> list[int]:
        """Return token IDs whose decoded text begins with char."""
        return self._by_first_char.get(char, [])

    def token_string(self, token_id: int) -> str:
        """Return the raw tokenizer spelling for an ID."""
        try:
            return self._id_to_token[token_id]
        except KeyError as e:
            raise VocabError(f"unknown token name: {token_id!r}") from e

    def token_text(self, token_id: int) -> str:
        """Convert Qwen whitespace markers to their emitted text."""
        raw = self.token_string(token_id)
        if raw.startswith("Ġ"):
            raw = raw.removeprefix("Ġ")
            raw = " " + raw
        elif raw == "Ċ":
            raw = "\n"
        return raw

    def id_of(self, raw_token: str) -> int | None:
        """Return the ID of a raw token, if present."""
        if raw_token in self.raw:
            return self.raw[raw_token]
        else:
            return None

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return len(self._id_to_token)

    def iter_tokens(self) -> Iterator[tuple[int, str]]:
        """Iterate over token ID and raw token pairs."""
        return iter(self._id_to_token.items())

    def string_value_token_ids(self, escape_pending: bool = False) -> set[int]:
        """Return cached IDs valid in the current JSON string escape state."""
        cached = self._string_value_ids_cache.get(escape_pending)
        if cached is not None:
            return cached
        result: set[int] = set()
        for token_id, _token in self.iter_tokens():
            text = self.token_text(token_id)
            if not text:
                continue
            if escape_pending:
                if len(text) == 1 and text in '"\\/bfnrt':
                    result.add(token_id)
            elif text == '"':
                result.add(token_id)
            elif "\n" not in text and "\\" not in text and '"' not in text:
                result.add(token_id)
        self._string_value_ids_cache[escape_pending] = result
        return result
