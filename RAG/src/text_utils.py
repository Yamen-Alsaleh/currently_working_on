import re
from pathlib import Path

from .models import Chunk


_WORD_RE: re.Pattern[str] = re.compile(r"\w+")
_CAMEL_RE: re.Pattern[str] = re.compile(
    r"[A-Z]+(?![a-z])\d*|[A-Z]?[a-z]+\d*|\d+"
)


def _split_identifier(piece: str) -> list[str]:
    parts = []
    for segment in piece.split("_"):
        if segment:
            segment_parts = _CAMEL_RE.findall(segment)
            if "".join(segment_parts).lower() != segment.lower():
                segment_parts = [segment]
            parts.extend(segment_parts)
    return [part.lower() for part in parts if part]


def tokenize(text: str) -> list[str]:
    tokens = []
    for piece in _WORD_RE.findall(text):
        whole = piece.lower()
        parts = _split_identifier(piece)
        if not parts:
            continue
        if parts == [whole]:
            tokens.append(whole)
        else:
            tokens.extend([whole, *parts])
    return tokens


def tokenize_chunk(chunk: Chunk, corpus_root: Path) -> list[str]:
    try:
        relative_path = chunk.file_path.relative_to(corpus_root)
    except ValueError:
        relative_path = chunk.file_path

    path_tokens = tokenize(str(relative_path))
    symbol_tokens = (
        tokenize(chunk.symbol_name)
        if chunk.symbol_name and chunk.symbol_name != "None"
        else []
    )
    text_tokens = tokenize(chunk.text)
    return path_tokens + symbol_tokens + text_tokens
