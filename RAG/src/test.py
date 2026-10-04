import ast
from pathlib import Path


def chunk_file(file_path: Path):
    source = read_source(file_path)
    offset_map = build_offset_map(source)
    tree = parse_ast(source)
    print(ast.dump(tree, indent=4))
    print(f"\n{offset_map}")


def read_source(file_path: Path) -> str:
    with file_path.open("r", encoding="utf-8") as file:
        content = file.read()
    return content


def build_offset_map(source_code: str) -> list[int]:
    lines = source_code.splitlines(keepends=True)
    offset_map = []
    total = 0
    for line in lines:
        offset_map.append(total)
        total += len(line)
    offset_map.append(len(source_code))
    return offset_map


def parse_ast(source_code: str) -> ast.Module:
    return ast.parse(source_code)


chunk_file(Path("./models.py"))
