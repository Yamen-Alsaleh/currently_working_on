from pathlib import Path

from pydantic import TypeAdapter, ValidationError
from .models import Chunk, IndexMetadata
from rank_bm25 import BM25Okapi
import pickle

INDEXED_EXTENSIONS = frozenset([".py", ".md"])
CHUNKS_FILE = "chunks.json"
BM25_FILE = "bm25.pkl"
META_FILE = "meta.json"


class IndexingError(Exception):
    """An Error for any unexpected errors that might happen during indexing"""


def discover_files(
    raw_dir: Path, extensions: frozenset[str] = INDEXED_EXTENSIONS
) -> list[Path]:
    """Find supported source files recursively under a corpus directory.

    Args:
        raw_dir: Root directory to search.
        extensions: File suffixes accepted for indexing.

    Returns:
        Sorted paths for supported files,
        excluding repository/cache directories.
    """
    result = []

    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() in extensions:
            result.append(path)
    result.sort()
    return result


def save_index(
    processed_dir: Path,
    chunks: list[Chunk],
    bm25: BM25Okapi,
    meta: IndexMetadata,
) -> None:
    """Persist chunk records, the BM25 object, and index metadata.

    Args:
        processed_dir: Destination directory for index artifacts.
        chunks: Chunk records aligned with the BM25 corpus order.
        bm25: Built BM25 index.
        meta: Metadata describing the index configuration and size.

    Raises:
        IndexingError: If the index artifacts cannot be written.
    """
    try:
        processed_dir.joinpath(META_FILE).unlink(missing_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        with open(processed_dir.joinpath(CHUNKS_FILE), "wb") as chunk_file:
            adapter = TypeAdapter(list[Chunk])
            chunks_json = adapter.dump_json(chunks)
            chunk_file.write(chunks_json)
        with open(processed_dir.joinpath(BM25_FILE), "wb") as bm_file:
            pickle.dump(bm25, bm_file)
        with open(
            processed_dir.joinpath(META_FILE), "w", encoding="utf-8"
        ) as meta_file:
            meta_json = meta.model_dump_json()
            meta_file.write(meta_json)
    except (OSError, pickle.PicklingError) as error:
        raise IndexingError("The index couldn't be written!") from error


def load_index(
    processed_dir: Path,
) -> tuple[list[Chunk], BM25Okapi, IndexMetadata]:
    """Load and validate all persisted index artifacts.

    Args:
        processed_dir: Directory containing the saved index.

    Returns:
        Chunks, the BM25 index, and index metadata in a tuple.

    Raises:
        IndexingError: If files are missing, corrupt, or inconsistent.
    """
    full_path = processed_dir / META_FILE
    if not full_path.is_file():
        raise IndexingError("no index was found")
    try:
        with open(full_path, "r", encoding="utf-8") as meta_file:
            meta = IndexMetadata.model_validate_json(meta_file.read())
        with open(processed_dir / CHUNKS_FILE, "rb") as chunk_file:
            adapter = TypeAdapter(list[Chunk])
            chunks = adapter.validate_json(chunk_file.read())
        with open(processed_dir / BM25_FILE, "rb") as bm_file:
            bm = pickle.load(bm_file)
        if not isinstance(bm, BM25Okapi):
            raise IndexingError("BM loading is not of the correct type")
    except (OSError, ValidationError) as error:
        raise IndexingError(
            f"index is corrupt and should be rebuilt{error}"
        ) from error
    except (
        pickle.UnpicklingError,
        EOFError,
        AttributeError,
        ImportError,
        UnicodeDecodeError,
        IndexError,
    ) as err:
        raise IndexingError(
            f"error while unpickling or reading {err}"
        ) from err

    if not (len(chunks) == meta.chunk_count == bm.corpus_size):
        raise IndexingError("the three index are out of sync")
    return (chunks, bm, meta)
