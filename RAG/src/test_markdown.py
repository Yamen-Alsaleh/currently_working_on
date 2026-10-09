"""Ad-hoc verification script for MarkdownChunker. Not graded, not committed."""

import sys
from pathlib import Path

from markdown_chunker import MarkdownChunker
from python_chunker import ChunkingError

RAW_DIR = Path("data/raw/vllm-0.10.1")
MAX_CHUNK_SIZE = 2000
OVERLAP_SIZE = 100


def check_file(path: Path, chunker: MarkdownChunker) -> list[str]:
    """Run one file through the chunker and return a list of problems."""
    problems: list[str] = []

    try:
        source = path.read_text(encoding="utf-8")
        chunks = chunker.chunk_file(path)
    except ChunkingError as error:
        return [f"{path}: chunking failed -> {error}"]

    if not source.strip():
        return problems  # empty/whitespace-only file: nothing to check

    if not chunks:
        return [f"{path}: produced zero chunks on a non-empty file"]

    ranges: list[tuple[int, int]] = []

    for chunk in chunks:
        start, end = chunk.first_character_index, chunk.last_character_index

        if end - start > MAX_CHUNK_SIZE:
            problems.append(
                f"{path}: chunk {chunk.symbol_name!r} size {end - start} "
                f"> max {MAX_CHUNK_SIZE} (span {start}:{end})"
            )

        if start < 0 or end > len(source) or start >= end:
            problems.append(
                f"{path}: chunk {chunk.symbol_name!r} has invalid span "
                f"{start}:{end} (file length {len(source)})"
            )
            continue

        if source[start:end] not in chunk.text:
            problems.append(
                f"{path}: chunk {chunk.symbol_name!r} span {start}:{end} "
                f"does not match its own text"
            )

        ranges.append((start, end))

    # coverage: unlike Python files, EVERY character of a markdown file
    # should belong to some chunk, so any gap is a real finding
    ranges.sort()
    merged_end = 0
    for start, end in ranges:
        if start > merged_end:
            snippet = source[merged_end:start][:40].replace("\n", "\\n")
            problems.append(
                f"{path}: uncovered gap {merged_end}:{start} -> {snippet!r}"
            )
        merged_end = max(merged_end, end)
    if merged_end < len(source):
        snippet = source[merged_end:][:40].replace("\n", "\\n")
        problems.append(
            f"{path}: uncovered tail {merged_end}:{len(source)} -> {snippet!r}"
        )

    return problems


def main() -> int:
    if not RAW_DIR.exists():
        print(f"raw dir not found: {RAW_DIR}", file=sys.stderr)
        return 1

    chunker = MarkdownChunker(
        max_chunk_size=MAX_CHUNK_SIZE, overlap_size=OVERLAP_SIZE
    )
    md_files = sorted(RAW_DIR.rglob("*.md"))
    print(f"testing {len(md_files)} markdown files...")

    all_problems: list[str] = []
    for path in md_files:
        all_problems.extend(check_file(path, chunker))

    print(f"\n{len(all_problems)} problems found across {len(md_files)} files")
    with open("/tmp/md_problems.txt", "w") as f:
        f.write("\n".join(all_problems))
    for problem in all_problems[:50]:
        print(" -", problem)
    if len(all_problems) > 50:
        print(f" ... and {len(all_problems) - 50} more (see /tmp/md_problems.txt)")

    return 1 if all_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())