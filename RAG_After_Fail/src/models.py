from pydantic import BaseModel, Field
from pathlib import Path
import uuid


class MinimalSource(BaseModel):
    """Represent a source file span used as evidence or a retrieval result."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question that has not yet been answered."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with its reference sources and expected answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Contain the questions used for RAG retrieval and answer evaluation."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represent one question and the source spans retrieved for it."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represent retrieval results together with the generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Contain retrieval results for a dataset and
    the configured top-k value."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswers(BaseModel):
    """Contain dataset retrieval results with generated answers and top-k."""

    search_results: list[MinimalAnswer]
    k: int


class Chunk(BaseModel):
    """Represent an indexed source fragment with offsets
    and retrieval metadata."""

    chunk_id: int
    file_path: Path
    first_character_index: int
    last_character_index: int
    text: str
    symbol_name: str | None = None
    symbol_type: str | None = None
    language: str

    def to_minimal_source(self) -> MinimalSource:
        """Convert this internal chunk into the subject-defined source model.

        Returns:
            A MinimalSource with a path string and the chunk character offsets.
        """
        return MinimalSource(
            file_path=self.file_path.as_posix(),
            first_character_index=self.first_character_index,
            last_character_index=self.last_character_index,
        )


class IndexMetadata(BaseModel):
    """Record configuration and counts needed to describe a persisted index."""

    max_chunk_size: int
    overlap_size: int
    k1: float
    b: float
    corpus_root: str
    file_count: int
    chunk_count: int
