import ast
from pathlib import Path
from .models import Chunk


class ChunkingError(Exception):
    """Report an error that occurred during chunking."""


class AstParseError(ChunkingError):
    """Report an Error that happened during parsing the source code."""


class PythonChunker:
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

    def chunk_file(self, file_path: Path) -> list[Chunk]:
        source = self._read_source(file_path)
        offset_map = self._build_offset_map(source)
        tree = self._parse_ast(source)
        structures = self._extract_structure(tree.body)
        result = []
        for struct in structures:
            temp = self._process_structure(
                struct, source, offset_map, file_path
            )
            result.extend(temp)
        return result

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

    def _parse_ast(self, source_code: str) -> ast.Module:
        try:
            return ast.parse(source_code)
        except SyntaxError as error:
            raise AstParseError(f"AST parsing error {error}") from error

    def _extract_structure(
        self, tree: list[ast.stmt]
    ) -> list[tuple[str, ast.stmt | list[ast.stmt]]]:
        ls: list[tuple[str, ast.stmt | list[ast.stmt]]] = []
        pending_ls: list[ast.stmt] = []
        for node in tree:
            if isinstance(node, ast.ClassDef):
                if pending_ls:
                    ls.append(("module_level", pending_ls))
                    pending_ls = []
                ls.append(("class", node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if pending_ls:
                    ls.append(("module_level", pending_ls))
                    pending_ls = []
                ls.append(("function", node))
            else:
                pending_ls.append(node)
        if pending_ls:
            ls.append(("module_level", pending_ls))
        return ls

    def _span(
        self,
        node: ast.stmt,
        offset_map: list[int],
    ) -> tuple[int, int]:
        if (
            isinstance(
                node, (ast.AsyncFunctionDef, ast.FunctionDef, ast.ClassDef)
            )
            and node.decorator_list
        ):
            first_lineno = node.decorator_list[0].lineno
        else:
            first_lineno = node.lineno

        start = offset_map[first_lineno - 1]
        if node.end_lineno is None:
            raise ChunkingError(f"node {node!r} has no end_lineno")
        end = offset_map[node.end_lineno]
        return (start, end)

    def _process_structure(
        self,
        structure: tuple[str, ast.stmt | list[ast.stmt]],
        source: str,
        offset_map: list[int],
        file_path: Path,
    ) -> list[Chunk]:
        result = []
        if structure[0] == "function":
            node = structure[1]
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                raise ChunkingError(f"expected a function node, got {node!r}")
            start, end = self._span(node, offset_map)
            if end - start <= self.max_chunk_size:
                result.append(
                    self._create_chunk(
                        file_path, source, start, end, node.name, "function"
                    )
                )
            else:
                result.extend(
                    self._split_oversized(
                        file_path, source, start, end, node.name, "function"
                    )
                )
        elif structure[0] == "class":
            node = structure[1]
            if not isinstance(node, ast.ClassDef):
                raise ChunkingError(f"expected a class node, got {node!r}")
            start, end = self._span(node, offset_map)
            if end - start <= self.max_chunk_size:
                result.append(
                    self._create_chunk(
                        file_path, source, start, end, node.name, "class"
                    )
                )
            else:
                internal_nodes = self._extract_structure(node.body)
                internal_result = []
                class_name = node.name
                for n in internal_nodes:
                    internal_result.extend(
                        self._process_structure(
                            n, source, offset_map, file_path
                        )
                    )
                for chunk in internal_result:
                    chunk.symbol_name = f"{class_name}.{chunk.symbol_name}"
                result.extend(internal_result)
        elif structure[0] == "module_level":
            payload = structure[1]
            if not isinstance(payload, list):
                raise ChunkingError(
                    f"expected a list of statements, got {payload!r}"
                )
            start, _ = self._span(payload[0], offset_map)
            _, end = self._span(payload[-1], offset_map)
            if end - start <= self.max_chunk_size:
                result.append(
                    self._create_chunk(
                        file_path, source, start, end, "None", "module"
                    )
                )
            else:
                result.extend(
                    self._split_oversized(
                        file_path, source, start, end, "None", "module"
                    )
                )
        return result

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
            language="python",
        )

    def _split_oversized(
        self,
        file_path: Path,
        source: str,
        start: int,
        end: int,
        symbol_name: str,
        symbol_type: str,
        header: str | None = None,
    ) -> list[Chunk]:
        step = self.max_chunk_size - self.overlap_size
        cur = start
        chunks = []
        while cur < end:
            piece_end = min(cur + self.max_chunk_size, end)
            chunk = self._create_chunk(
                file_path, source, cur, piece_end, symbol_name, symbol_type
            )
            chunks.append(chunk)
            if piece_end == end:
                break
            cur += step
        return chunks
