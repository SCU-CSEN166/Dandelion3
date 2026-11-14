"""Adapter checks without model downloads or hidden-set answer labels."""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_hidden_tests as runner


def test_workbook_validation(monkeypatch):
    openpyxl = pytest.importorskip("openpyxl")
    rows = [("message_id", "audio_file", "transcript"), (17.0, "msg.mp4", " Report ")]
    sheet = SimpleNamespace(iter_rows=lambda **kwargs: iter(rows))
    closed = []
    book = SimpleNamespace(worksheets=[sheet], close=lambda: closed.append(True))
    monkeypatch.setattr(openpyxl, "load_workbook", lambda *args, **kwargs: book)
    assert runner.load_transcripts(Path("unused.xlsx")) == [{"id": 17, "text": "Report"}]
    rows.append((17, "msg2.mp4", "Repeated ID"))
    with pytest.raises(ValueError, match="duplicate message_id"):
        runner.load_transcripts(Path("unused.xlsx"))
    rows[-1] = (18, "msg2.mp4", "")
    with pytest.raises(ValueError, match="transcript must contain text"):
        runner.load_transcripts(Path("unused.xlsx"))
    assert len(closed) == 3


def test_model_handoff_and_csv(monkeypatch, tmp_path):
    messages = [
        {"id": 12, "text": "private"},
        {"id": 4, "text": "hello"},
        {"id": 29, "text": "private"},
    ]
    monkeypatch.setattr(runner, "anonymize_text", lambda text, model: text.replace("private", "[PERSON]"))

    def classify(items, model):
        assert [item["text"] for item in items] == ["[PERSON]", "hello", "[PERSON]"]
        return [{**item, "predicted_label": "irrelevant" if item["id"] == 4 else "relevant"}
                for item in items]

    monkeypatch.setattr(runner, "classify_relevance", classify)
    monkeypatch.setattr(runner, "extract_attributes", lambda text, date: {
        "activity_type": "training", "quantity": "18 women",
    })

    def dedup(events, model):
        assert [event["source_message_ids"] for event in events] == [[12], [29]]
        events[1]["duplicate_of"] = events[0]["event_id"]

    monkeypatch.setattr(runner, "find_duplicates", dedup)
    results = runner.run_hidden_tests(messages)
    output = tmp_path / "outputs" / "results.csv"
    runner.write_results(results, output)
    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == list(runner.FIELDS)
        rows = list(reader)
    assert [row["Message_id"] for row in rows] == ["12", "4", "29"]
    assert rows[0]["participants"] == "18"
    assert rows[0]["duplicate_of"] == "none"
    assert rows[1]["relevance"] == "Chit-chat"
    assert rows[1]["activity_type"] == rows[1]["participants"] == ""
    assert rows[2]["duplicate_of"] == "12"
    assert messages[0]["text"] == "private"


@pytest.mark.parametrize("quantity", [None, "2 goats", "18000 shillings", "forty-two women"])
def test_missing_or_nonparticipant_quantity_stays_blank(quantity):
    assert runner.participant_count(quantity) == ""
