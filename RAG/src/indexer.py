import os
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from file_utils import IndexingError, discover_files, save_index
from markdown_chunker import MarkdownChunker
from models import Chunk, IndexMetadata
from python_chunker import ChunkingError, PythonChunker
from text_utils import tokenize_chunk


class Indexer:
    def __init__(
        self,
        max_chunk_size: int = 2000,
        overlap_size: int = 100,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._python_chunker = PythonChunker(max_chunk_size, overlap_size)
        self._markdown_chunker = MarkdownChunker(max_chunk_size, overlap_size)
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.k1 = k1
        self.b = b

    def build(self, raw_dir: Path, processed_dir: Path) -> int:
        if not raw_dir.is_dir():
            raise IndexingError(
                f"raw_dir must exist and be a directory: {raw_dir}"
            )

        try:
            raw_dir = Path(os.path.abspath(raw_dir)).relative_to(Path.cwd())
        except ValueError as error:
            raise IndexingError(
                "raw_dir must be under the current directory; run from the project root"
            ) from error

        files = discover_files(raw_dir)
        if not files:
            raise IndexingError(f"no indexable files found in {raw_dir}")

        corpus_root = self._resolve_corpus_root(raw_dir)
        chunks, skipped = self._chunk_files(files)
        if not chunks:
            raise IndexingError(
                "no chunks could be created from the discovered files"
            )

        self._assign_ids(chunks)
        self._validate_chunks(chunks)
        tokenized = self._tokenize_chunks(chunks, corpus_root)
        bm25 = self._build_bm25(tokenized)
        meta = IndexMetadata(
            max_chunk_size=self.max_chunk_size,
            overlap_size=self.overlap_size,
            k1=self.k1,
            b=self.b,
            corpus_root=str(corpus_root),
            file_count=len(files),
            chunk_count=len(chunks),
        )
        save_index(processed_dir, chunks, bm25, meta)

        if skipped:
            skipped_names = ", ".join(str(path) for path in skipped[:5])
            print(f"Skipped {len(skipped)} files: {skipped_names}")

        return len(chunks)

    def _resolve_corpus_root(self, raw_dir: Path) -> Path:
        child_directories = [path for path in raw_dir.iterdir() if path.is_dir()]
        if len(child_directories) == 1:
            return child_directories[0]
        return raw_dir

    def _chunk_files(self, files: list[Path]) -> tuple[list[Chunk], list[Path]]:
        chunks: list[Chunk] = []
        skipped: list[Path] = []
        for file_path in tqdm(files, desc="Chunking", unit="file"):
            try:
                chunks.extend(self._chunk_one(file_path))
            except ChunkingError:
                skipped.append(file_path)
        return chunks, skipped

    def _chunk_one(self, file_path: Path) -> list[Chunk]:
        if file_path.suffix.lower() == ".py":
            return self._python_chunker.chunk_file(file_path)
        return self._markdown_chunker.chunk_file(file_path)

    def _assign_ids(self, chunks: list[Chunk]) -> None:
        for chunk_id, chunk in enumerate(chunks):
            chunk.chunk_id = chunk_id

    def _validate_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            span = chunk.last_character_index - chunk.first_character_index
            if (
                chunk.first_character_index < 0
                or span <= 0
                or span > self.max_chunk_size
            ):
                raise IndexingError(
                    f"invalid chunk in {chunk.file_path}: span {span} "
                    f"at {chunk.first_character_index}:{chunk.last_character_index}"
                )

    def _tokenize_chunks(
        self, chunks: list[Chunk], corpus_root: Path
    ) -> list[list[str]]:
        return [
            tokenize_chunk(chunk, corpus_root)
            for chunk in tqdm(chunks, desc="Tokenizing", unit="chunk")
        ]

    def _build_bm25(self, tokenized: list[list[str]]) -> BM25Okapi:
        if not tokenized:
            raise IndexingError("cannot build BM25 from an empty corpus")
        return BM25Okapi(tokenized, k1=self.k1, b=self.b)
