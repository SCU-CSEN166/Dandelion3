import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_full_pipeline_on_synthetic_data():
    """End-to-end check on the real synthetic dataset: segmentation should
    merge the two-message goat-distribution thread into one event, and
    dedup should catch the later independent confirmation of it."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    from pipeline import run_pipeline

    output = run_pipeline(ROOT / "data/synthetic/whatsapp_export.txt")

    events_by_ids = {tuple(e["source_message_ids"]): e for e in output["events"]}
    assert (3, 4) in events_by_ids, "messages 3+4 (same sender, 2 minutes apart) should merge into one event"

    goat_event = events_by_ids[(3, 4)]
    duplicate_events = [e for e in output["events"] if e["duplicate_of"] == goat_event["event_id"]]
    assert any(
        e["source_message_ids"] == [26] for e in duplicate_events
    ), "the later independent confirmation (message 26) should be flagged as a duplicate"

    assert goat_event["attributes"]["activity_type"] == "livelihood_distribution"
    assert goat_event["attributes"]["quantity"] == "2 goats"

    from relevance_classifier import parse_whatsapp_export

    n_messages = len(parse_whatsapp_export(ROOT / "data/synthetic/whatsapp_export.txt"))
    all_ids = {mid for e in output["events"] for mid in e["source_message_ids"]} | {
        m["message_id"] for m in output["irrelevant_messages"]
    }
    assert len(all_ids) == n_messages, "every source message should end up in exactly one bucket"
