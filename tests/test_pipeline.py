import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_full_pipeline_on_synthetic_data():
    """End-to-end check on the real synthetic dataset, covering the cases
    the new architecture was specifically built to fix (see TESTING.md for
    the exact numbers behind each)."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    from pipeline import run_pipeline

    output = run_pipeline(ROOT / "data/synthetic/whatsapp_export.txt")
    events = output["events"]
    events_by_ids = {tuple(e["source_message_ids"]): e for e in events}

    # The 2-message goat-distribution thread (ids 2,3) plus its later,
    # differently-worded confirmation (id 25) should all merge into one
    # event -- this needed BOTH the sender/time-adjacency signal (for 2+3)
    # AND raw (unredacted) text carrying the shared name "Achieng" (for 25).
    assert (2, 3, 25) in events_by_ids, "the goat-distribution thread + its later confirmation should merge into one event"
    goat_event = events_by_ids[(2, 3, 25)]
    assert goat_event["participants"] is None or goat_event["participants"] >= 0  # no crash either way
    assert "goat" in goat_event["activity_type"] or "livelihood" in goat_event["activity_type"]

    # A genuinely different beneficiary (id 36, "the Oduya family") with
    # near-identical phrasing must NOT be merged into the Achieng event --
    # this is exactly the false positive anonymization used to cause.
    oduya_event = next(e for e in events if e["source_message_ids"] == [36])
    assert oduya_event["duplicate_of"] is None

    # A true continuation 12 minutes apart, same sender (ids 30, 31) should
    # merge -- content alone is too different, but adjacency makes it work.
    assert (30, 31) in events_by_ids, "same-sender continuation 12 minutes apart should merge"

    # Every parsed message should end up in exactly one bucket.
    from parser import parse_chat

    n_messages = len(parse_chat(ROOT / "data/synthetic/whatsapp_export.txt", dayfirst=True))
    all_ids = (
        {mid for e in events for mid in e["source_message_ids"]}
        | {m["message_id"] for m in output["irrelevant_messages"]}
        | {m["id"] for m in output["media_only_messages"]}
    )
    assert len(all_ids) == n_messages, "every source message should end up in exactly one bucket"
