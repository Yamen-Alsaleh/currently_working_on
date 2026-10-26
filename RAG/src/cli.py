from pathlib import Path
from typing import Any

from .evaluation import compute_recall
from .file_utils import IndexingError
from pydantic import ValidationError, BaseModel
import sys
from .models import (
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswers,
)
from .retriever import Retriever
from .python_chunker import ChunkingError
from .indexer import Indexer
from generator import Generator, GenerationError

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_SEARCH_DIR = "data/output/search_results"
DEFAULT_ANSWER_DIR = "data/output/search_results_and_answer"
MAX_ALLOWED_CHUNK_SIZE = 2000


class CLIError(Exception):
    """Catches any error that happens during the CLI with clear messages"""


EXPECTED_ERRORS = (
    CLIError,
    IndexingError,
    ChunkingError,
    ValidationError,
    OSError,
    UnicodeDecodeError,
    GenerationError,
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


def _check_k(k: int) -> None:
    if not isinstance(k, int) or k <= 0:
        raise CLIError("k must be a positive integer!")


def _check_query(query: Any) -> str:
    q = str(query).strip()
    if not q:
        raise CLIError("query must not be empty!")
    return q


def _load_dataset(dataset_path: str) -> RagDataset:
    path = Path(dataset_path)
    if path.is_file():
        data = path.read_text(encoding="utf-8")
        return RagDataset.model_validate_json(data)
    raise CLIError(f"dataset file not found: {dataset_path}")


def _write_json(directory: str, filename: str, model: BaseModel) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / filename).write_text(
        model.model_dump_json(indent=2), encoding="utf-8"
    )
    return path / filename


def _load_search_results(path: str) -> StudentSearchResults:
    data_path = Path(path)
    if data_path.is_file():
        data = data_path.read_text(encoding="utf-8")
        return StudentSearchResults.model_validate_json(data)
    raise CLIError(f"search results file not found: {data_path}")


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
            if not isinstance(k1, (int, float)) or not isinstance(
                b, (int, float)
            ):
                raise CLIError("k1 and b must be numbers")
            if k1 < 0 or b < 0 or b > 1:
                raise CLIError("k1 must be >= 0 and b must be >= 0 and <= 1")
            indexer = Indexer(max_chunk_size, overlap_size, k1, b)
            total_chunks = indexer.build(Path(raw_dir), Path(processed_dir))
            print(
                f"Ingestion complete! Indexed {total_chunks}"
                f" chunks under {processed_dir}"
            )
        except EXPECTED_ERRORS as error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)

    def search(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        try:
            query = _check_query(query)
            _check_k(k)
            chunks = Retriever(Path(processed_dir)).search(query, k)
            if not chunks:
                print("no results found!")
                return
            for chunk in chunks:
                source = chunk.to_minimal_source()
                print(
                    f"{source.file_path} [{source.first_character_index}"
                    f":{source.last_character_index}]"
                )
        except EXPECTED_ERRORS as error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = DEFAULT_SEARCH_DIR,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        try:
            _check_k(k)
            dataset = _load_dataset(dataset_path)
            retriever = Retriever(Path(processed_dir))
            results = retriever.search_batch(dataset.rag_questions, k)
            output = StudentSearchResults(search_results=results, k=k)
            path = _write_json(save_directory, Path(dataset_path).name, output)
            print(f"Saved student_search_results to {path}")
        except EXPECTED_ERRORS as error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        try:
            student_result = _load_search_results(student_search_results_path)
            truth = _load_dataset(dataset_path)
            for k in (1, 3, 5, 10):
                if k <= student_result.k:
                    recall = compute_recall(student_result, truth, k)
                    print(f"Recall@{k}: {recall:.3f} ({recall * 100:.1f%})")
        except EXPECTED_ERRORS as error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)

    def answer(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        try:
            query = _check_query(query)
            _check_k(k)
            chunks = Retriever(Path(processed_dir)).search(query, k)
            if not chunks:
                print("no results found!")
                return
            sources = [chunk.to_minimal_source() for chunk in chunks]
            answer = Generator().generate(query, sources)
            print(answer)
            print("Sources:\n")
            for i, source in enumerate(sources, start=1):
                first = source.first_character_index
                last = source.last_character_index
                path = source.file_path
                with open(path, "r", encoding="utf-8") as file:
                    text = file.read()
                    print(f"Source {i} ({path}): {text[first:last]}\n")
        except EXPECTED_ERRORS as error:
            print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = DEFAULT_ANSWER_DIR,
    ) -> None:
        student_results = _load_search_results(student_search_results_path)
        answers = Generator().answer_batch(student_results.search_results)
        model = StudentSearchResultsAndAnswers(
            search_results=answers, k=student_results.k
        )
        _write_json(save_directory, save_directory, model)
        print(f"Saved student_search_results_and_answer to {save_directory}")
