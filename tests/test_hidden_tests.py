"""Adapter checks without model downloads or hidden-set answer labels."""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_hidden_tests as runner
import activity_review


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


def test_audio_mode_feeds_whisper_text_to_pipeline(monkeypatch, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    workbook_path = tmp_path / "transcripts.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.append(("message_id", "audio_file", "transcript"))
    workbook.active.append((17, "msg02.mp4", "Stored transcript"))
    workbook.active.append((4, "msg01.mp4", None))
    workbook.save(workbook_path)
    workbook.close()

    audio_dir = tmp_path / "recordings"
    audio_dir.mkdir()
    for name in ("msg01.mp4", "msg02.mp4"):
        (audio_dir / name).touch()

    models = []

    def make_transcriber(model):
        models.append(model)

        def transcribe(path):
            assert Path(path).parent == audio_dir.resolve()
            return f"Whisper text for {Path(path).name}"

        return SimpleNamespace(transcribe=transcribe)

    def classify(messages, model):
        assert messages == [
            {"id": 17, "text": "Whisper text for msg02.mp4"},
            {"id": 4, "text": "Whisper text for msg01.mp4"},
        ]
        return [{**message, "predicted_label": "irrelevant"} for message in messages]

    monkeypatch.setattr(runner, "Transcriber", make_transcriber)
    monkeypatch.setattr(runner, "classify_relevance", classify)
    monkeypatch.setattr(runner, "anonymize_text", lambda text, model: text)
    output = tmp_path / "results.csv"
    monkeypatch.setattr(sys, "argv", [
        "run_hidden_tests.py", "--input", str(workbook_path), "--audio-dir", str(audio_dir),
        "--asr-model", "test-whisper", "--output", str(output),
    ])

    runner.main()

    assert models == ["test-whisper"]
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["Message_id"] for row in rows] == ["17", "4"]
    assert [row["anonymized_text"] for row in rows] == [
        "Whisper text for msg02.mp4", "Whisper text for msg01.mp4",
    ]


@pytest.mark.parametrize("merge_reports", [False, True])
def test_model_handoff_and_csv(monkeypatch, tmp_path, merge_reports):
    messages = [
        {"id": 12, "text": "private"},
        {"id": 4, "text": "hello"},
        {"id": 29, "text": "private"},
    ]
    anonymize = lambda text, model: text.replace("private", "[PERSON]")
    monkeypatch.setattr(runner, "anonymize_text", anonymize)
    monkeypatch.setattr(activity_review, "anonymize_text", anonymize)

    def classify(items, model):
        assert [item["text"] for item in items] == ["private", "hello", "private"]
        return [{**item, "predicted_label": "irrelevant" if item["id"] == 4 else "relevant", "confidence": 0.94}
                for item in items]

    monkeypatch.setattr(runner, "classify_relevance", classify)
    monkeypatch.setattr(runner, "extract_attributes", lambda text, date: {
        "activity_type": "training", "participants": 18,
    })

    def cluster(items, model):
        assert [item["text"] for item in items] == ["private", "private"]
        assert all(item["participants"] == 18 for item in items)
        groups = [items] if merge_reports else [[item] for item in items]
        return [{
            "source_message_ids": [item["id"] for item in group],
            "text": " ".join(item["text"] for item in group),
            "activity_type": "training", "participants": 18,
            "relevance_confidence": 0.94,
        } for group in groups]

    monkeypatch.setattr(runner, "cluster_events", cluster)

    def dedup(events, model):
        assert all("private" in event["text"] for event in events)
        if not merge_reports:
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
    assert rows[0]["anonymized_text"] == "[PERSON]"
    assert rows[0]["duplicate_of"] == "none"
    assert rows[1]["relevance"] == "Chit-chat"
    assert rows[1]["activity_type"] == rows[1]["participants"] == ""
    assert rows[2]["duplicate_of"] == "12"
    assert messages[0]["text"] == "private"


@pytest.mark.parametrize("count, expected", [(0, 0), (None, "")])
def test_zero_and_unknown_counts(monkeypatch, count, expected):
    monkeypatch.setattr(runner, "classify_relevance", lambda items, model: [
        {**items[0], "predicted_label": "relevant", "confidence": 0.94},
    ])
    monkeypatch.setattr(runner, "extract_attributes", lambda text, date: {
        "activity_type": "training", "participants": count,
    })
    monkeypatch.setattr(runner, "anonymize_text", lambda text, model: text)
    monkeypatch.setattr(activity_review, "anonymize_text", lambda text, model: text)
    rows = runner.run_hidden_tests([{"id": 1, "text": "training report"}])
    assert rows[0]["participants"] == expected
