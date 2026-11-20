"""
Week 6 baseline: relevance filtering (pipeline step 4 in the Week 5 plan).

Classifies each WhatsApp message as either a real field-activity report
(relevant) or casual chit-chat (irrelevant), using zero-shot classification
so no labeled training data is required yet.

Usage:
    python src/relevance_classifier.py \
        --input data/synthetic/whatsapp_export.txt \
        --output outputs/relevance_predictions.jsonl \
        --gold data/synthetic/gold_labels.jsonl
"""

import argparse
import json
import re
from pathlib import Path

if __package__:
    from .extract import NUMBER_TOKEN, PARTICIPANT_NOUNS
else:
    from extract import NUMBER_TOKEN, PARTICIPANT_NOUNS

MESSAGE_RE = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{4}),\s*(?P<time>\d{1,2}:\d{2})\s*-\s*"
    r"(?P<sender>[^:]+):\s*(?P<text>.*)$"
)

CANDIDATE_LABELS = [
    "a report of a field activity such as a home visit, training session, "
    "or distribution of supplies",
    "casual conversation, small talk, or group chit-chat",
]
RELEVANT_LABEL = CANDIDATE_LABELS[0]

_BORDERLINE_CHIT_CHAT = 0.60
_ACTIVITY_RE = re.compile(
    r"\b(?:table[ -]banking|savings[ -]group|meeting|training|outreach|workshop|session)\b",
    re.IGNORECASE,
)
_ATTENDANCE_RE = re.compile(
    rf"\b{NUMBER_TOKEN}\s+(?:{PARTICIPANT_NOUNS})\s+(?:attended|participated)\b",
    re.IGNORECASE,
)


def parse_whatsapp_export(path: Path) -> list[dict]:
    messages = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = MESSAGE_RE.match(line)
            if not match:
                continue
            messages.append(
                {
                    "id": len(messages) + 1,
                    "date": match.group("date"),
                    "time": match.group("time"),
                    "sender": match.group("sender").strip(),
                    "text": match.group("text").strip(),
                }
            )
    return messages


def classify_relevance(messages: list[dict], model_name: str) -> list[dict]:
    from transformers import pipeline

    classifier = pipeline("zero-shot-classification", model=model_name)
    results = []
    for msg in messages:
        text = msg["text"]
        output = classifier(text, candidate_labels=CANDIDATE_LABELS)
        predicted = "relevant" if output["labels"][0] == RELEVANT_LABEL else "irrelevant"
        confidence = output["scores"][0]
        if (
            predicted == "irrelevant"
            and confidence < _BORDERLINE_CHIT_CHAT
            and _ACTIVITY_RE.search(text)
            and _ATTENDANCE_RE.search(text)
        ):
            predicted = "relevant"
            # Keep BART's event score so the review step can still flag the close call.
            confidence = output["scores"][1]
        results.append({**msg, "predicted_label": predicted, "confidence": round(confidence, 4)})
    return results


def evaluate(predictions: list[dict], gold_path: Path) -> dict:
    gold = {}
    with open(gold_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gold[row["id"]] = row["label"]

    tp = fp = tn = fn = 0
    for pred in predictions:
        truth = gold.get(pred["id"])
        if truth is None:
            continue
        predicted_relevant = pred["predicted_label"] == "relevant"
        actually_relevant = truth == "relevant"
        if predicted_relevant and actually_relevant:
            tp += 1
        elif predicted_relevant and not actually_relevant:
            fp += 1
        elif not predicted_relevant and not actually_relevant:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Relevance-filter synthetic WhatsApp messages.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--model", default="facebook/bart-large-mnli")
    args = parser.parse_args()

    messages = parse_whatsapp_export(args.input)
    print(f"Parsed {len(messages)} messages from {args.input}")

    predictions = classify_relevance(messages, args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"Wrote predictions to {args.output}")

    if args.gold:
        metrics = evaluate(predictions, args.gold)
        print(f"Evaluation against {args.gold}: {metrics}")


if __name__ == "__main__":
    main()
