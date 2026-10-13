from pathlib import Path
from typing import Sequence
from tqdm import tqdm
from models import Chunk, MinimalSearchResults, UnansweredQuestion
from file_utils import load_index
from text_utils import tokenize


class retriever:
    def __init__(self, processed_dir: Path) -> None:
        self.chunks, self.bm25, self.meta = load_index(processed_dir)

    def search(self, query: str, k: int) -> list[Chunk]:
        if k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        indices = self._rank(tokens, k)
        return [self.chunks[i] for i in indices]

    def _rank(self, query_tokens: list[str], k: int) -> list[int]:
        scores = self.bm25.get_scores(query_tokens)
        indices = scores.argsort()[::-1]
        return indices.tolist()[: min(k, len(self.chunks))]

    def search_batch(
        self, questions: Sequence[UnansweredQuestion], k: int
    ) -> list[MinimalSearchResults]:
        result = []
        for question in tqdm(questions):
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
