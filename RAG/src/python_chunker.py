import ast
from pathlib import Path
from models import Chunk


class InputError(Exception):
    """Report an invalid or unreadable prompt input file."""


class PythonChunker:
    max_chunk_size: int
    overlap_size: int

    def __init__(
        self, max_chunk_size: int = 2000, overlap_size: int = 100
    ) -> None:
        if max_chunk_size > 2000 or max_chunk_size <= 0:
            raise ValueError(
                "max_chunk_size should be a positive integer and <= 2000"
            )
        self.max_chunk_size = max_chunk_size
        if overlap_size >= max_chunk_size:
            raise ValueError("overlap_size should be less than max_chunk_size")
        self.overlap_size = overlap_size

    def chunk_file(self, file_path: Path) -> list[Chunk]:
        source = self._read_source(file_path)
        offset_map = self._build_offset_map(source)

    def _read_source(self, file_path: Path) -> str:
        try:
            with file_path.open("r", encoding="utf-8") as file:
                content = file.read()
        except (OSError, UnicodeDecodeError) as error:
            raise InputError(f"could not read {file_path}: {error}") from error
        return content

    def _build_offset_map(self, source_code: str) -> list[int]:
        pass

    def _parse_ast(self):
        pass

    def _extract_structure(self):
        pass

    def _process_structure(self):
        pass

    def _create_chunk(self):
        pass

    def _split_oversized(self):
        pass
