"""
Event clustering -- replaces the old two-step design (time/sender
segmentation, then a separate post-hoc embedding-only dedup check) with one
pass over every relevant message in the whole chat, not just adjacent ones
from the same sender.

Runs on raw (unredacted) text. That matters: with names/places redacted,
two *different* real events can become nearly textually identical (the
only thing that told them apart was the name), while two *real* duplicates
in different words score low on pure embedding similarity. Running on raw
text, and combining it with agreement on extracted attributes
(participants, activity type), gives real separation where text
similarity alone doesn't -- see TESTING.md for the actual numbers this
was tuned against.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

_STOPWORD_LEN = 3  # skip short/common words when checking activity-type overlap


@lru_cache(maxsize=1)
def _get_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _activity_overlaps(a: str | None, b: str | None) -> bool:
    if not a or not b or a == "other" or b == "other":
        # "other" means extraction found nothing, not that the two
        # messages share a topic -- don't credit it as a match.
        return False
    # The old fixed-category labels are underscore_joined; free-text ones
    # are space-separated. Split on both so e.g. "livelihood_distribution"
    # can still overlap with "...the livelihood program".
    words_a = {w for w in a.lower().replace("_", " ").split() if len(w) > _STOPWORD_LEN}
    words_b = {w for w in b.lower().replace("_", " ").split() if len(w) > _STOPWORD_LEN}
    return bool(words_a & words_b)


def _same_sender_nearby(msg_a: dict, msg_b: dict, max_gap_minutes: float) -> bool:
    """Same sender, close together in time -- a strong signal for a
    continuation message ("forgot to mention X") that has little content
    overlap with the message it's continuing, so text/activity/participant
    signals alone would miss it."""
    sender_a, sender_b = msg_a.get("sender"), msg_b.get("sender")
    ts_a, ts_b = msg_a.get("timestamp"), msg_b.get("timestamp")
    if not sender_a or sender_a != sender_b or not ts_a or not ts_b:
        return False
    try:
        gap = abs((datetime.fromisoformat(ts_a) - datetime.fromisoformat(ts_b)).total_seconds()) / 60
    except ValueError:
        return False
    return gap <= max_gap_minutes


def _combined_score(
    msg_a: dict,
    msg_b: dict,
    text_similarity: float,
    text_weight: float,
    activity_weight: float,
    participants_weight: float,
    adjacency_weight: float,
    max_gap_minutes: float,
) -> float:
    score = text_similarity * text_weight
    if _activity_overlaps(msg_a.get("activity_type"), msg_b.get("activity_type")):
        score += activity_weight
    pa, pb = msg_a.get("participants"), msg_b.get("participants")
    if pa is not None and pb is not None and pa == pb:
        score += participants_weight
    if _same_sender_nearby(msg_a, msg_b, max_gap_minutes):
        score += adjacency_weight
    return score


def _union_find(n: int):
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    return find, union


def cluster_events(
    messages: list[dict],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    threshold: float = 0.5,
    text_weight: float = 0.5,
    activity_weight: float = 0.15,
    participants_weight: float = 0.15,
    adjacency_weight: float = 0.3,
    max_gap_minutes: float = 15,
) -> list[dict]:
    """`messages`: relevant messages only, each with at least
    `id`, `text`, `activity_type`, `participants`. Returns a list of event
    dicts: `{source_message_ids, text, activity_type, participants}`."""
    if not messages:
        return []

    if len(messages) > 1:
        embedder = _get_embedder(embedding_model)
        texts = [m["text"] for m in messages]
        embeddings = embedder.encode(texts, normalize_embeddings=True)
    else:
        embeddings = None

    n = len(messages)
    find, union = _union_find(n)

    for i in range(n):
        for j in range(i + 1, n):
            text_similarity = float(embeddings[i] @ embeddings[j]) if embeddings is not None else 0.0
            score = _combined_score(
                messages[i], messages[j], text_similarity, text_weight, activity_weight,
                participants_weight, adjacency_weight, max_gap_minutes,
            )
            if score >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    return [_aggregate_event([messages[i] for i in idxs]) for idxs in groups.values()]


def _aggregate_event(cluster_msgs: list[dict]) -> dict:
    activity_type = next(
        (m["activity_type"] for m in cluster_msgs if m.get("activity_type") and m["activity_type"] != "other"),
        cluster_msgs[0].get("activity_type"),
    )
    participant_values = [m["participants"] for m in cluster_msgs if m.get("participants") is not None]
    return {
        "source_message_ids": sorted(m["id"] for m in cluster_msgs),
        "text": " ".join(m["text"] for m in cluster_msgs),
        "activity_type": activity_type,
        "participants": max(participant_values) if participant_values else None,
        "relevance_confidence": max(m.get("relevance_confidence", 0.0) for m in cluster_msgs),
    }
