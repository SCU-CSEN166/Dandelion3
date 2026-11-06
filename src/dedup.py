"""
Deduplication (pipeline step 7): detect events that describe the same
underlying activity, even when reported by a different sender at a
different time (so segmentation's sender+time rule won't have caught it).

The Week 5 plan already proposed sentence-transformers MiniLM for this step
as an alternative to an LLM judgment call; we're using it directly as the
baseline since a local Llama setup isn't ready yet (see README).
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def find_duplicates(
    events: list[dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    threshold: float = 0.75,
) -> None:
    """Mutates `events` in place, setting duplicate_of / duplicate_confidence."""
    for event in events:
        event.setdefault("duplicate_of", None)
        event.setdefault("duplicate_confidence", None)

    if len(events) < 2:
        return

    embedder = _get_embedder(model_name)
    texts = [event["anonymized_text"] for event in events]
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    for i in range(len(events)):
        if events[i]["duplicate_of"] is not None:
            continue
        for j in range(i + 1, len(events)):
            if events[j]["duplicate_of"] is not None:
                continue
            score = float(embeddings[i] @ embeddings[j])
            if score >= threshold:
                events[j]["duplicate_of"] = events[i]["event_id"]
                events[j]["duplicate_confidence"] = round(score, 4)
