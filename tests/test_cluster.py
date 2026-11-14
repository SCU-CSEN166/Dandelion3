import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cluster import _activity_overlaps, cluster_events


def test_activity_overlap_handles_underscore_vs_space():
    # Fixed-category labels are underscore_joined; free-text ones are
    # space-separated -- make sure overlap detection bridges that.
    assert _activity_overlaps("livelihood_distribution", "gave 2 goats as part of the livelihood program")
    assert not _activity_overlaps("training", "cervical cancer screening")
    assert not _activity_overlaps(None, "training")


def test_activity_overlap_does_not_credit_mutual_other():
    # "other" means extraction found nothing -- two unrelated messages that
    # both failed extraction must not get an "activity matches" bonus.
    assert not _activity_overlaps("other", "other")


def test_empty_and_single_message():
    assert cluster_events([]) == []
    events = cluster_events([{"id": 1, "text": "Visited the homestead.", "activity_type": "other", "participants": None}])
    assert len(events) == 1
    assert events[0]["source_message_ids"] == [1]


def test_clusters_real_duplicate_pair_from_different_senders():
    pytest.importorskip("sentence_transformers")
    messages = [
        {
            "id": 1,
            "text": "This is Grace, the CHV from Rongai. We served 42 women on family planning today.",
            "activity_type": "family planning outreach",
            "participants": 42,
        },
        {
            "id": 2,
            "text": "Good evening, it's Faith. I was also at the outreach today with Grace -- 42 women served on family planning.",
            "activity_type": "the outreach",
            "participants": 42,
        },
        {
            "id": 3,
            "text": "Today we did the cervical cancer screening camp. We screened 28 women.",
            "activity_type": "cervical cancer screening camp",
            "participants": 28,
        },
    ]
    events = cluster_events(messages)
    assert len(events) == 2
    ids_by_event = {tuple(e["source_message_ids"]) for e in events}
    assert (1, 2) in ids_by_event
    assert (3,) in ids_by_event


def test_does_not_merge_different_events_with_same_participant_count():
    pytest.importorskip("sentence_transformers")
    messages = [
        {"id": 1, "text": "Delivered mosquito nets to 20 households in Shibuye.", "activity_type": "distribution", "participants": 20},
        {"id": 2, "text": "Ran a hygiene training for 20 schoolchildren at the primary school.", "activity_type": "training", "participants": 20},
    ]
    events = cluster_events(messages)
    assert len(events) == 2
