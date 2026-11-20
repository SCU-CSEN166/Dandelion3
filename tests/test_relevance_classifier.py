"""
Tests for the Week 6 relevance classifier.

The parser, scoring, and fallback tests run without model downloads. The
integration test checks the zero-shot model on our synthetic dataset.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.relevance_classifier import classify_relevance, evaluate, parse_whatsapp_export


def test_parse_whatsapp_export(tmp_path):
    sample = (
        "03/02/2026, 08:41 - Amina K.: Visited Mama Achieng's homestead today, gave 2 goats.\n"
        "03/02/2026, 08:52 - Grace W.: lol did anyone see the match last night\n"
    )
    export = tmp_path / "sample.txt"
    export.write_text(sample)

    messages = parse_whatsapp_export(export)

    assert len(messages) == 2
    assert messages[0] == {
        "id": 1,
        "date": "03/02/2026",
        "time": "08:41",
        "sender": "Amina K.",
        "text": "Visited Mama Achieng's homestead today, gave 2 goats.",
    }
    assert messages[1]["sender"] == "Grace W."


def test_evaluate_perfect_predictions(tmp_path):
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": 1, "label": "relevant"},
                {"id": 2, "label": "irrelevant"},
                {"id": 3, "label": "relevant"},
            ]
        )
    )
    predictions = [
        {"id": 1, "predicted_label": "relevant"},
        {"id": 2, "predicted_label": "irrelevant"},
        {"id": 3, "predicted_label": "relevant"},
    ]

    metrics = evaluate(predictions, gold)

    assert metrics == {
        "tp": 2, "fp": 0, "tn": 1, "fn": 0,
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
    }


def test_evaluate_catches_false_positive_and_false_negative(tmp_path):
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": 1, "label": "relevant"},
                {"id": 2, "label": "irrelevant"},
            ]
        )
    )
    predictions = [
        {"id": 1, "predicted_label": "irrelevant"},  # false negative
        {"id": 2, "predicted_label": "relevant"},  # false positive
    ]

    metrics = evaluate(predictions, gold)

    assert metrics["tp"] == 0
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


@pytest.mark.parametrize(
    ("text", "chit_chat_score", "expected_label", "expected_score"),
    [
        ("We held a savings group meeting. Twenty-five members attended.", 0.55, "relevant", 0.45),
        ("We held an outreach session. 18 women participated.", 0.60, "irrelevant", 0.60),
        ("Eighteen people signed up for tomorrow's training.", 0.55, "irrelevant", 0.55),
        ("Twenty-five people attended the birthday party.", 0.55, "irrelevant", 0.55),
    ],
)
def test_borderline_chit_chat_requires_activity_and_attendance(
    monkeypatch, text, chit_chat_score, expected_label, expected_score,
):
    def classify(_text, candidate_labels):
        return {"labels": candidate_labels[::-1], "scores": [chit_chat_score, 1 - chit_chat_score]}

    transformers = SimpleNamespace(pipeline=lambda *args, **kwargs: classify)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    [result] = classify_relevance([{"id": 1, "text": text}], "test-model")

    assert result["predicted_label"] == expected_label
    assert result["confidence"] == expected_score


def test_real_dataset_meets_f1_threshold():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    messages = parse_whatsapp_export(ROOT / "data/synthetic/whatsapp_export.txt")
    predictions = classify_relevance(messages, "facebook/bart-large-mnli")
    metrics = evaluate(predictions, ROOT / "data/synthetic/gold_labels.jsonl")

    assert metrics["f1"] >= 0.8, f"F1 dropped below the Week 5 target: {metrics}"
