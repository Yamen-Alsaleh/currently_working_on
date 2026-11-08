from pathlib import Path
from typing import Sequence
from tqdm import tqdm
from .models import Chunk, MinimalSearchResults, UnansweredQuestion
from .file_utils import load_index
from .text_utils import tokenize


class Retriever:
    """Load a persisted index and expose single-query and batch retrieval."""

    def __init__(self, processed_dir: Path) -> None:
        """Load chunks, BM25 data, and metadata from the processed directory.

        Args:
            processed_dir: Directory containing the saved index artifacts.
        """
        self.chunks, self.bm25, self.meta = load_index(processed_dir)

    def search(self, query: str, k: int) -> list[Chunk]:
        """Retrieve the highest-ranked chunks for a query.

        Args:
            query: Natural-language or code-oriented search text.
            k: Maximum number of chunks to return.

        Returns:
            Ranked chunks, or an empty list for invalid k or tokenless queries.
        """
        if k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        indices = self._rank(tokens, k)
        return [self.chunks[i] for i in indices]

    def _rank(self, query_tokens: list[str], k: int) -> list[int]:
        """Rank chunk indices by BM25 score in descending order.

        Args:
            query_tokens: Tokenized query.
            k: Maximum number of ranked indices to return.

        Returns:
            Chunk indices ordered from highest to lowest score.
        """
        scores = self.bm25.get_scores(query_tokens)
        indices = (-scores).argsort(kind="stable")
        result: list[int] = indices[: min(k, len(self.chunks))].tolist()
        return result

    def search_batch(
        self, questions: Sequence[UnansweredQuestion], k: int
    ) -> list[MinimalSearchResults]:
        """Run retrieval for each question while reporting progress.

        Args:
            questions: Questions to search for.
            k: Maximum number of sources to retrieve per question.

        Returns:
            Search-result models in the same order as the input questions.
        """
        result = []
        for question in tqdm(questions, desc="Searching", unit="question"):
            chunks = self.search(question.question, k)
            result.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=[
                        chunk.to_minimal_source() for chunk in chunks
                    ],
                )
            )
        return result
