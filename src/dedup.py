"""
Deduplication safety net: runs AFTER cluster.py has already grouped
messages into events. Clustering does the main work of merging messages
about the same event; this is a second pass over the resulting EVENTS,
in case two separate event clusters still turn out to describe the same
real-world activity (e.g. two groups that individually fell just short of
the clustering threshold). Sets `duplicate_of` at the event level, as
requested, so two whole events can be collapsed together, not just two
messages.

Runs on raw (unredacted) text, same as clustering -- see cluster.py's
docstring for why that matters.
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
    texts = [event["text"] for event in events]
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
