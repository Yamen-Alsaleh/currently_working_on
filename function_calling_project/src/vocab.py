from pathlib import Path
from typing import Iterator
from pydantic import BaseModel, PrivateAttr, model_validator, ValidationError
import json


class VocabError(Exception):
    pass


class TokenVocab(BaseModel):
    raw: dict[str, int]
    _id_to_token: dict[int, str] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def build(self):
        for token, id in self.raw.items():
            if id in self._id_to_token:
                raise ValueError(f"duplicate token: {token!r}")
            else:
                self._id_to_token[id] = token
        return self

    @classmethod
    def load(cls, path: str | Path) -> "TokenVocab":
        try:
            with open(path, "r") as file:
                text = file.read()
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as e:
                    raise VocabError(f"{path} is not valid JSON: {e}") from e

                if not isinstance(parsed, dict):
                    raise VocabError(
                        f"{path} must contain a JSON array,"
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

    def token_string(self, token_id: int) -> str:
        try:
            return self._id_to_token[token_id]
        except KeyError as e:
            raise VocabError(f"unknown token name: {token_id!r}") from e

    def token_text(self, token_id: int) -> str:
        raw = self.token_string(token_id)
        if raw.startswith("Ġ"):
            raw = raw.removeprefix("Ġ")
            raw = " " + raw
        return raw

    def id_of(self, raw_token: str) -> int | None:
        if raw_token in self.raw:
            return self.raw[raw_token]
        else:
            return None

    def __len__(self) -> int:
        return len(self._id_to_token)

    def __iter__(self) -> Iterator[tuple[int, str]]:
        return iter(self._id_to_token.items())
