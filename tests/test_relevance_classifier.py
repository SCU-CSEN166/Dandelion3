"""
Tests for the Week 6 relevance classifier.

The first three tests are fast and need no model/network access — they check
the WhatsApp parser and the scoring logic directly. The last test is a real
integration check: it runs the actual zero-shot model on our synthetic
dataset and asserts the F1 we report in the README still holds.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.relevance_classifier import evaluate, parse_whatsapp_export


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


def test_real_dataset_meets_f1_threshold():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from src.relevance_classifier import classify_relevance

    messages = parse_whatsapp_export(ROOT / "data/synthetic/whatsapp_export.txt")
    predictions = classify_relevance(messages, "facebook/bart-large-mnli")
    metrics = evaluate(predictions, ROOT / "data/synthetic/gold_labels.jsonl")

    assert metrics["f1"] >= 0.8, f"F1 dropped below the Week 5 target: {metrics}"
