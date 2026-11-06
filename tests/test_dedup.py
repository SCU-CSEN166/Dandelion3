import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_near_duplicate_events_flagged():
    pytest.importorskip("sentence_transformers")
    from dedup import find_duplicates

    events = [
        {
            "event_id": "evt_0001",
            "anonymized_text": "Visited [PERSON]'s homestead today, gave 2 goats as part of the livelihood program.",
        },
        {
            "event_id": "evt_0002",
            "anonymized_text": "Conducted a nutrition training session for 14 mothers at the community hall.",
        },
        {
            "event_id": "evt_0003",
            "anonymized_text": "Also confirming [PERSON]'s homestead received the 2 goats from the livelihood program today.",
        },
    ]

    find_duplicates(events)

    assert events[0]["duplicate_of"] is None
    assert events[1]["duplicate_of"] is None
    assert events[2]["duplicate_of"] == "evt_0001"
    assert events[2]["duplicate_confidence"] > 0.7


def test_single_event_not_flagged():
    from dedup import find_duplicates

    events = [{"event_id": "evt_0001", "anonymized_text": "Visited the homestead."}]
    find_duplicates(events)
    assert events[0]["duplicate_of"] is None
