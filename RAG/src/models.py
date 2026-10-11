from pydantic import BaseModel, Field
from pathlib import Path
import uuid


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswers(BaseModel):
    search_results: list[MinimalAnswer]
    k: int


class Chunk(BaseModel):
    chunk_id: int
    file_path: Path
    first_character_index: int
    last_character_index: int
    text: str
    symbol_name: str | None = None
    symbol_type: str | None = None
    language: str


class IndexMetadata(BaseModel):
    max_chunk_size: int
    overlap_size: int
    k1: float
    b: float
    corpus_root: str
    file_count: int
    chunk_count: int
