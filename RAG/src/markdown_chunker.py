from dataclasses import dataclass
import re
from models import Chunk
from python_chunker import ChunkingError
from pathlib import Path


@dataclass
class _Heading:
    level: int
    title: str
    line_number: int
    start: int = 0
    end: int = 0


class MarkdownChunker:
    max_chunk_size: int
    overlap_size: int

    def __init__(
        self, max_chunk_size: int = 2000, overlap_size: int = 100
    ) -> None:
        if max_chunk_size <= 0:
            raise ChunkingError("max_chunk_size must be a positive integer!")
        self.max_chunk_size = max_chunk_size
        if overlap_size >= max_chunk_size or overlap_size < 0:
            raise ChunkingError(
                "overlap_size should be positive integer less"
                " than max_chunk_size"
            )
        self.overlap_size = overlap_size

    def chunk_file(self, file_path: Path) -> list[Chunk]: ...

    def _read_source(self, file_path: Path) -> str:
        try:
            with file_path.open("r", encoding="utf-8") as file:
                content = file.read()
        except (OSError, UnicodeDecodeError) as error:
            raise ChunkingError(
                f"could not read {file_path}: {error}"
            ) from error
        return content

    def _build_offset_map(self, source_code: str) -> list[int]:
        lines = source_code.splitlines(keepends=True)
        offset_map = []
        total = 0
        for line in lines:
            offset_map.append(total)
            total += len(line)
        offset_map.append(len(source_code))
        return offset_map

    def _find_headings(self, source_code: str) -> list[_Heading]:
        lines = source_code.splitlines()
        in_fence = False
        headings: list[_Heading] = []
        line_number = 1
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            level = 0
            if re.match(r"^#{1,6} ", line):
                for char in line:
                    if char == "#":
                        level += 1
                    elif char == " ":
                        break
                raw_title = line.strip("# ")
                headings.append(
                    _Heading(
                        level=level, title=raw_title, line_number=line_number
                    )
                )
        return headings

    def _compute_heading_spans(
        self, headings: list[_Heading], offset_map: list[int], source_len: int
    ) -> tuple[tuple[int, int], list[_Heading]]:
        if not headings:
            return ((0, source_len), headings)
        first_start = offset_map[headings[0].line_number - 1]
        preamble_span = (0, first_start)
        for i in range(len(headings)):
            headings[i].start = offset_map[headings[i].line_number - 1]
            j = i + 1
            flag = False
            while j < len(headings):
                if headings[j].level <= headings[i].level:
                    headings[i].end = offset_map[headings[j].line_number - 1]
                    flag = True
            if not flag:
                headings[i].end = source_len
        return (preamble_span, headings)

    def _top_level_indices(self, headings: list[_Heading]) -> list[int]:
        if not headings:
            return []
        else:
            min_level = min(h.level for h in headings)
            return [i for i, h in enumerate(headings) if h.level == min_level]

    def _find_children(
        self, parent_index: int, headings: list[_Heading]
    ) -> list[int]:
        parent = headings[parent_index]
        j = parent_index + 1
        childs_indices = []
        while j < len(headings):
            if (
                headings[j].start < parent.end
                and headings[j].level < parent.level
            ):
                childs_indices.append(j)
            j += 1
        return childs_indices

    def _process_section(
        self,
        index: int,
        headings: list[_Heading],
        source: str,
        offset_map: list[int],
        file_path: Path,
    ) -> list[Chunk]:
        heading = headings[index]
        start, end = heading.start, heading.end
        if end - start <= self.max_chunk_size:
            return [
                self._create_chunk(
                    file_path, source, start, end, heading.title, "section"
                )
            ]
        children = self._find_children(index, headings)
        result = []
        intro_end = headings[children[0]].start if children else end
        if intro_end > start:
            if intro_end - start <= self.max_chunk_size:
                result.append(
                    self._create_chunk(
                        file_path,
                        source,
                        start,
                        intro_end,
                        heading.title,
                        "section",
                    )
                )
            else:
                result.extend(
                    self._split_paragraphs(
                        start,
                        intro_end,
                        source,
                        heading.title,
                        "section",
                        file_path,
                    )
                )

        for child_index in children:
            result.extend(
                self._process_section(
                    child_index, headings, source, offset_map, file_path
                )
            )

        return result

    def _split_paragraphs(
        self,
        start: int,
        end: int,
        source: str,
        symbol_name: str | None,
        symbol_type: str,
        file_path: Path,
        header: str | None = None,
    ) -> list[Chunk]:
        boundaries = []

    def _create_chunk(
        self,
        file_path: Path,
        source: str,
        start: int,
        end: int,
        symbol_name: str,
        symbol_type: str,
        header: str | None = None,
    ) -> Chunk:
        if header:
            body = header + source[start:end]
        else:
            body = source[start:end]
        return Chunk(
            chunk_id=0,
            file_path=file_path,
            first_character_index=start,
            last_character_index=end,
            text=body,
            symbol_name=symbol_name,
            symbol_type=symbol_type,
            language="markdown",
        )
