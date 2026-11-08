import hashlib
from pathlib import Path
from typing import Sequence

from .models import Chunk, UnansweredQuestion
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .models import Chunk, MinimalSearchResults, UnansweredQuestion
from .file_utils import load_index
from .text_utils import tokenize


class Retriever:
    """Retrieve relevant chunks using BM25 and dense search with RRF."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    RRF_K = 60
    CANDIDATE_SIZE = 100

    def __init__(self, processed_dir: Path) -> None:
        """Load the existing index and initialize dense retrieval."""
        self.processed_dir = processed_dir
        self.chunks, self.bm25, self.meta = load_index(processed_dir)

        self.model = SentenceTransformer(self.MODEL_NAME)

        self.embeddings_path = processed_dir / "dense_embeddings.npy"
        self.fingerprint_path = processed_dir / "dense_embeddings.sha256"

        self.embeddings = self._load_or_build_embeddings()

    def search_batch(
        self,
        questions: Sequence[UnansweredQuestion],
        k: int,
    ) -> list[MinimalSearchResults]:
        """Retrieve relevant sources for a batch of questions.

        Args:
            questions: Questions to retrieve sources for.
            k: Maximum number of sources to retrieve per question.

        Returns:
            Search results containing each question and its retrieved sources.
        """
        results = []

        for question in tqdm(questions, desc="Searching questions"):
            chunks = self.search(question.question, k)

            results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=[
                        chunk.to_minimal_source() for chunk in chunks
                    ],
                )
            )

        return results

    def search(self, query: str, k: int) -> list[Chunk]:
        """Return the top-k chunks using BM25 and dense retrieval."""
        if k <= 0 or not query.strip():
            return []

        query_tokens = list(dict.fromkeys(tokenize(query)))
        if not query_tokens:
            return []

        bm25_ranking = self._bm25_rank(query_tokens)
        dense_ranking = self._dense_rank(query)

        fused_ranking = self._rrf(bm25_ranking, dense_ranking)

        return [self.chunks[i] for i in fused_ranking[:k]]

    def _embedding_text(self, chunk: Chunk) -> str:
        """Build the text representation used for dense embeddings."""
        symbol = chunk.symbol_name or ""
        return f"File: {chunk.file_path}\nSymbol: {symbol}\n\n{chunk.text}"

    def _fingerprint(self) -> str:
        """Calculate a fingerprint to detect changes in indexed chunks."""
        digest = hashlib.sha256()

        for chunk in self.chunks:
            digest.update(self._embedding_text(chunk).encode("utf-8"))

        return digest.hexdigest()

    def _load_or_build_embeddings(self) -> np.ndarray:
        """Load cached embeddings or generate and save them."""
        fingerprint = self._fingerprint()

        if self.embeddings_path.exists() and self.fingerprint_path.exists():
            try:
                saved_fingerprint = self.fingerprint_path.read_text().strip()
                embeddings = np.load(self.embeddings_path)

                if saved_fingerprint == fingerprint and embeddings.shape[
                    0
                ] == len(self.chunks):
                    return embeddings

            except (OSError, ValueError):
                pass

        texts = [self._embedding_text(chunk) for chunk in self.chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        temporary_path = self.embeddings_path.with_suffix(".tmp")

        with temporary_path.open("wb") as file:
            np.save(file, embeddings)

        temporary_path.replace(self.embeddings_path)

        self.fingerprint_path.write_text(fingerprint)

        return embeddings

    def _bm25_rank(self, query_tokens: list[str]) -> list[int]:
        """Return the top BM25 candidate indices."""
        scores = self.bm25.get_scores(query_tokens)

        indices = np.argsort(-scores, kind="stable")

        limit = min(self.CANDIDATE_SIZE, len(self.chunks))

        return indices[:limit].tolist()

    def _dense_rank(self, query: str) -> list[int]:
        """Return the top semantic-search candidate indices."""
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores = self.embeddings @ query_embedding

        indices = np.argsort(-scores, kind="stable")

        limit = min(self.CANDIDATE_SIZE, len(self.chunks))

        return indices[:limit].tolist()

    def _rrf(
        self,
        bm25_ranking: list[int],
        dense_ranking: list[int],
    ) -> list[int]:
        """Combine two rankings using Reciprocal Rank Fusion."""
        scores: dict[int, float] = {}

        for ranking in (bm25_ranking, dense_ranking):
            for rank, chunk_index in enumerate(ranking, start=1):
                scores[chunk_index] = scores.get(chunk_index, 0.0) + (
                    1.0 / (self.RRF_K + rank)
                )

        return sorted(scores, key=lambda index: scores[index], reverse=True)
