import re
from pathlib import Path

from .models import Chunk


_WORD_RE: re.Pattern[str] = re.compile(r"\w+")
_CAMEL_RE: re.Pattern[str] = re.compile(
    r"[A-Z]+(?![a-z])\d*|[A-Z]?[a-z]+\d*|\d+"
)


def _split_identifier(piece: str) -> list[str]:
    """Break an identifier into underscore- and CamelCase-derived terms.

    Args:
        piece: One word-like piece extracted from text.

    Returns:
        Lowercase identifier components.
    """
    parts = []
    for segment in piece.split("_"):
        if segment:
            segment_parts = _CAMEL_RE.findall(segment)
            if "".join(segment_parts).lower() != segment.lower():
                segment_parts = [segment]
            parts.extend(segment_parts)
    return [part.lower() for part in parts if part]


def tokenize(text: str) -> list[str]:
    """Tokenize text and retain both complete identifiers and their parts.

    Args:
        text: Source code, documentation, path, or query text.

    Returns:
        Lowercase lexical tokens suitable for BM25 indexing or querying.
    """
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
    """Tokenize a chunk together with its relative path and symbol metadata.

    Args:
        chunk: Chunk whose searchable terms should be assembled.
        corpus_root: Root directory used to make the chunk path relative.

    Returns:
        Combined path, symbol, and content tokens for BM25.
    """
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
