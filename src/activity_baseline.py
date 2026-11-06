"""Process Roland's anonymized JSON records. Run with: input.json output.json"""

import json
import sys
from datetime import date


FIELDS = ("activity_type", "date", "location", "quantity", "outcome")


def clean_message(message):
    # Use the same fields for every record, with null for missing values.
    attributes = {}
    for field in FIELDS:
        value = message["attributes"].get(field)
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{field} must be a string or null")
            value = " ".join(value.split()) or None
        attributes[field] = value

    if attributes["activity_type"]:
        attributes["activity_type"] = attributes["activity_type"].lower().replace(" ", "_")
    if attributes["date"]:
        attributes["date"] = date.fromisoformat(attributes["date"]).isoformat()

    confidence = message["relevance"]["confidence"]
    if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
        raise ValueError("Relevance confidence must be a number between 0 and 1")
    source_ids = message["source_message_ids"]
    if not source_ids or any(type(item) is not int for item in source_ids):
        raise ValueError("Each message needs integer source_message_ids")

    return {
        "source_message_ids": sorted(set(source_ids)),
        "anonymized_text": message["anonymized_text"].strip(),
        "relevance": message["relevance"].copy(),
        "attributes": attributes,
        "activity_key": message.get("activity_key"),
    }


def combine_messages(messages, event_id):
    attributes = {}
    conflicts = {}
    reasons = []
    for field in FIELDS:
        # Keep disagreements for review instead of picking one message's value.
        values = []
        for message in messages:
            value = message["attributes"][field]
            if value is not None and value not in values:
                values.append(value)
        attributes[field] = values[0] if len(values) == 1 else None
        if len(values) > 1:
            conflicts[field] = values
            reasons.append(f"Conflicting {field}")

    missing = [field for field in FIELDS if attributes[field] is None]
    if missing:
        reasons.append("Missing fields: " + ", ".join(missing))
    relevance = min(message["relevance"]["confidence"] for message in messages)
    # This is a baseline review score, not a probability that the extraction is correct.
    score = relevance
    if missing:
        score = min(score, 0.6)
    if conflicts:
        score = min(score, 0.4)
    if relevance < 0.7:
        reasons.append("Low relevance confidence")

    # Keep the original message IDs so we can trace an activity back to its sources.
    source_ids = []
    texts = []
    for message in messages:
        source_ids.extend(message["source_message_ids"])
        if message["anonymized_text"] not in texts:
            texts.append(message["anonymized_text"])
    return {
        "event_id": event_id,
        "source_message_ids": sorted(set(source_ids)),
        "anonymized_text": "\n".join(texts),
        "relevance": {"label": "relevant", "confidence": relevance},
        "attributes": attributes,
        "duplicate_of": None,
        "duplicate_confidence": None,
        "conflicts": conflicts,
        "confidence": {"score": score, "needs_review": bool(reasons), "reasons": reasons},
    }


def process_messages(messages):
    groups = {}
    seen = []
    for message in messages:
        if message["relevance"]["label"] == "irrelevant":
            continue
        if message["relevance"]["label"] != "relevant":
            raise ValueError("Relevance label must be relevant or irrelevant")
        message = clean_message(message)
        # Skip repeated records after cleaning up their formatting.
        if message in seen:
            continue
        seen.append(message)

        # A shared activity_key links messages; without one, keep them separate.
        key = message["activity_key"] or tuple(message["source_message_ids"])
        groups.setdefault(key, []).append(message)

    events = []
    for messages in groups.values():
        event = combine_messages(messages, f"evt_{len(events) + 1:04d}")
        events.append(event)
    return events


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 src/activity_baseline.py input.json output.json")
    with open(sys.argv[1], encoding="utf-8") as file:
        messages = json.load(file)
    events = process_messages(messages)
    with open(sys.argv[2], "w", encoding="utf-8") as file:
        json.dump(events, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Saved {len(events)} activities")
