from pathlib import Path
from .file_utils import IndexingError
from pydantic import ValidationError
import sys
from .python_chunker import ChunkingError
from .indexer import Indexer

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_SEARCH_DIR = "data/output/search_results"
MAX_ALLOWED_CHUNK_SIZE = 2000


class CLIError(Exception):
    """Catches any error that happens during the CLI with clear messages"""


EXPECTED_ERRORS = (
    CLIError,
    IndexingError,
    ChunkingError,
    ValidationError,
    OSError,
)


def _check_chunk_params(max_chunk_size: int, overlap_size: int) -> None:
    if (
        not isinstance(max_chunk_size, int)
        or not max_chunk_size > 0
        or max_chunk_size > MAX_ALLOWED_CHUNK_SIZE
    ):
        raise CLIError(
            "max_chunk_size must be a positive integer"
            " and less than or equal to 2000"
        )

    if (
        not isinstance(overlap_size, int)
        or overlap_size < 0
        or overlap_size >= max_chunk_size
    ):
        raise CLIError(
            "overlap_size must be a non negative integer"
            f" and less than max_chunk_size:{max_chunk_size}"
        )


class CLI:
    def index(
        self,
        max_chunk_size: int = 2000,
        overlap_size: int = 100,
        raw_dir: str = DEFAULT_RAW_DIR,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        try:
            _check_chunk_params(max_chunk_size, overlap_size)
            if k1 < 0 or b < 0 or b > 1:
                raise CLIError("k1 must be >= 0 and b must be > 0 and < 1")
            indexer = Indexer(max_chunk_size, overlap_size, k1, b)
            total_chunks = indexer.build(Path(raw_dir), Path(processed_dir))
            print(
                f"Ingestion complete! Indexed {total_chunks}"
                f" chunks under {processed_dir}"
            )
        except EXPECTED_ERRORS as error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
