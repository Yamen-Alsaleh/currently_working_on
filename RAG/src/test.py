"""Ad-hoc verification script for PythonChunker. Not graded, not committed."""

import sys
from pathlib import Path

from python_chunker import PythonChunker, ChunkingError

RAW_DIR = Path("data/raw/vllm-0.10.1")
MAX_CHUNK_SIZE = 2000
OVERLAP_SIZE = 100


def check_file(path: Path, chunker: PythonChunker) -> list[str]:
    """Run one file through the chunker and return a list of problem strings."""
    problems: list[str] = []

    try:
        source = path.read_text(encoding="utf-8")
        chunks = chunker.chunk_file(path)
    except ChunkingError as error:
        return [f"{path}: chunking failed -> {error}"]

    if not chunks:
        problems.append(f"{path}: produced zero chunks")
        return problems

    ranges: list[tuple[int, int]] = []

    for chunk in chunks:
        start, end = chunk.first_character_index, chunk.last_character_index

        # max size
        if end - start > MAX_CHUNK_SIZE:
            problems.append(
                f"{path}: chunk {chunk.symbol_name!r} size {end - start} "
                f"> max {MAX_CHUNK_SIZE} (span {start}:{end})"
            )

        # bounds sanity
        if start < 0 or end > len(source) or start > end:
            problems.append(
                f"{path}: chunk {chunk.symbol_name!r} has invalid span "
                f"{start}:{end} (file length {len(source)})"
            )
            continue

        # slice equality: the reported span must equal what's actually
        # in chunk.text (allowing for a header prefix, if any)
        expected = source[start:end]
        if expected not in chunk.text:
            problems.append(
                f"{path}: chunk {chunk.symbol_name!r} span {start}:{end} "
                f"does not match its own text"
            )

        ranges.append((start, end))

    # coverage: merged ranges should reconstruct the whole file length
    ranges.sort()
    merged_end = 0
    gaps: list[tuple[int, int]] = []
    for start, end in ranges:
        if start > merged_end:
            gaps.append((merged_end, start))
        merged_end = max(merged_end, end)
    if merged_end < len(source):
        gaps.append((merged_end, len(source)))

    for gap_start, gap_end in gaps:
        if gap_end - gap_start > 0:
            snippet = source[gap_start:gap_end][:40].replace("\n", "\\n")
            problems.append(
                f"{path}: uncovered gap {gap_start}:{gap_end} -> {snippet!r}"
            )

    return problems


def main() -> int:
    if not RAW_DIR.exists():
        print(f"raw dir not found: {RAW_DIR}", file=sys.stderr)
        return 1

    chunker = PythonChunker(
        max_chunk_size=MAX_CHUNK_SIZE, overlap_size=OVERLAP_SIZE
    )
    py_files = sorted(RAW_DIR.rglob("*.py"))
    print(f"testing {len(py_files)} python files...")

    total_chunks = 0
    total_problems: list[str] = []

    for path in py_files:
        problems = check_file(path, chunker)
        if problems:
            total_problems.extend(problems)
        else:
            total_chunks += len(chunker.chunk_file(path))

    print(
        f"\n{len(total_problems)} problems found across {len(py_files)} files"
    )
    with open("/tmp/all_problems.txt", "w") as f:
        f.write("\n".join(total_problems))
    for problem in total_problems[:50]:
        print(" -", problem)
    if len(total_problems) > 50:
        print(f" ... and {len(total_problems) - 50} more")

    return 1 if total_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
