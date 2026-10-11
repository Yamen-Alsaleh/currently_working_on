from pathlib import Path

from pydantic import TypeAdapter
from models import Chunk, IndexMetadata
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
    try:
        processed_dir.joinpath(META_FILE).unlink(missing_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)
        with open(processed_dir.joinpath(CHUNKS_FILE), "wb") as chunk_file:
            adapter = TypeAdapter(list[Chunk])
            data = adapter.dump_json(chunks)
            chunk_file.write(data)
        with open(processed_dir.joinpath(BM25_FILE), "wb") as bm_file:
            pickle.dump(bm25, bm_file)
        with open(
            processed_dir.joinpath(META_FILE), "w", encoding="utf-8"
        ) as meta_file:
            data2 = meta.model_dump_json()
            meta_file.write(data2)
    except (OSError, pickle.PicklingError) as error:
        raise IndexingError("The index couldn't be written!") from error
