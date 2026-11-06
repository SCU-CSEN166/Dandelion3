import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from segment import segment_events


def _msg(id_, sender, date, time, text):
    return {"id": id_, "sender": sender, "date": date, "time": time, "anonymized_text": text}


def test_merges_same_sender_within_gap():
    messages = [
        _msg(3, "Amina K.", "03/02/2026", "08:41", "Visited the homestead, gave 2 goats."),
        _msg(4, "Amina K.", "03/02/2026", "08:43", "Also left some pamphlets."),
    ]
    events = segment_events(messages)
    assert len(events) == 1
    assert events[0]["source_message_ids"] == [3, 4]
    assert "pamphlets" in events[0]["anonymized_text"]


def test_does_not_merge_different_sender():
    messages = [
        _msg(3, "Amina K.", "03/02/2026", "08:41", "Visited the homestead, gave 2 goats."),
        _msg(10, "Lilian N.", "03/02/2026", "08:42", "Delivered mosquito nets."),
    ]
    events = segment_events(messages)
    assert len(events) == 2


def test_does_not_merge_outside_time_window():
    messages = [
        _msg(3, "Amina K.", "03/02/2026", "08:41", "Visited the homestead, gave 2 goats."),
        _msg(13, "Amina K.", "03/02/2026", "10:30", "Unrelated later message."),
    ]
    events = segment_events(messages, max_gap_minutes=10)
    assert len(events) == 2


def test_empty_input():
    assert segment_events([]) == []
