from .models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


def sources_match(retrieved: MinimalSource, truth: MinimalSource) -> bool:
    """Determine whether retrieved and expected source spans overlap enough.

    Args:
        retrieved: Source span returned by retrieval.
        truth: Expected source span from the reference dataset.

    Returns:
        True when the paths match and span overlap meets the threshold.
    """
    if retrieved.file_path != truth.file_path:
        return False
    overlap = min(
        retrieved.last_character_index, truth.last_character_index
    ) - max(retrieved.first_character_index, truth.first_character_index)
    if overlap <= 0:
        return False
    union = (
        (retrieved.last_character_index - retrieved.first_character_index)
        + (truth.last_character_index - truth.first_character_index)
        - overlap
    )
    return overlap / union >= 0.05


def compute_recall(
    student: StudentSearchResults, truth: RagDataset, k: int
) -> float:
    """Compute mean per-question source recall at the requested cutoff.

    Args:
        student: Student retrieval results and their configured cutoff.
        truth: Reference dataset containing expected source spans.
        k: Number of top retrieved sources to evaluate.

    Returns:
        Mean recall across answerable questions with expected sources.
    """
    by_id: dict[str, MinimalSearchResults] = {}
    for result in student.search_results:
        by_id[result.question_id] = result
    recalls: list[float] = []
    for question in truth.rag_questions:
        if not isinstance(question, AnsweredQuestion) or not question.sources:
            continue
        student_result = by_id.get(question.question_id)
        if student_result is None:
            retrieved: list[MinimalSource] = []
        else:
            retrieved = student_result.retrieved_sources[:k]
        found = 0
        for correct in question.sources:
            if any(sources_match(r, correct) for r in retrieved):
                found += 1
        recalls.append(found / len(question.sources))
    if not recalls:
        return 0.0
    return sum(r for r in recalls) / len(recalls)
