"""
Full mid-pipeline baseline covering Roland's scope: anonymize -> relevance
filter -> event segmentation -> attribute extraction -> deduplication.

Built on top of Sathvik's ingestion step (`parser.py`) and ASR step
(`asr.py`) rather than a separate parser, so this is the real end-to-end
system, not a standalone demo. Produces the shared JSON handoff format for
Ronav's formatting/testing step.

Usage:
    python src/pipeline.py \
        --input data/synthetic/whatsapp_export.txt \
        --output outputs/pipeline_output.json
"""

import argparse
import json
from pathlib import Path

from anonymize import anonymize_text
from dedup import find_duplicates
from extract import extract_attributes
from parser import Message, parse_chat
from relevance_classifier import classify_relevance
from segment import segment_events


def _message_text(m: Message) -> str:
    """Prefer the ASR transcript for voice notes; fall back to the raw text."""
    return m.transcript if (m.is_voice_note and m.transcript) else m.text


def run_pipeline(
    input_path: Path,
    relevance_model: str = "facebook/bart-large-mnli",
    ner_model: str = "dslim/bert-base-NER",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    dayfirst: bool = True,
) -> dict:
    raw_messages = parse_chat(input_path, dayfirst=dayfirst)
    messages = [
        {"id": m.msg_id, "timestamp": m.timestamp, "sender": m.sender, "text": _message_text(m)}
        for m in raw_messages
        if not m.is_system
    ]
    print(f"Parsed {len(messages)} messages from {input_path} (via Sathvik's parser.py)")

    # Step 3: anonymize every message before anything downstream sees it.
    for msg in messages:
        msg["anonymized_text"] = anonymize_text(msg["text"], ner_model)

    # Step 4: relevance-filter on the anonymized text (the classifier never
    # needs to see raw PII).
    classifier_input = [{**msg, "text": msg["anonymized_text"]} for msg in messages]
    classified = classify_relevance(classifier_input, relevance_model)
    for msg, result in zip(messages, classified):
        msg["predicted_label"] = result["predicted_label"]
        msg["relevance_confidence"] = result["confidence"]

    relevant_messages = [m for m in messages if m["predicted_label"] == "relevant"]
    irrelevant_messages = [m for m in messages if m["predicted_label"] == "irrelevant"]

    # Step 5: group relevant messages into events.
    events = segment_events(relevant_messages)

    messages_by_id = {m["id"]: m for m in messages}
    for event in events:
        source_date = event["_first_timestamp"].split("T")[0]
        event["attributes"] = extract_attributes(event["anonymized_text"], source_date)
        event["relevance"] = {
            "label": "relevant",
            "confidence": max(
                messages_by_id[mid]["relevance_confidence"] for mid in event["source_message_ids"]
            ),
        }
        del event["sender"]
        del event["_first_timestamp"]

    # Step 7: flag events that describe the same underlying activity.
    find_duplicates(events, embedding_model)

    return {
        "events": events,
        "irrelevant_messages": [
            {
                "message_id": m["id"],
                "anonymized_text": m["anonymized_text"],
                "confidence": m["relevance_confidence"],
            }
            for m in irrelevant_messages
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run the Roland-scope mid-pipeline end-to-end.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--relevance-model", default="facebook/bart-large-mnli")
    parser.add_argument("--ner-model", default="dslim/bert-base-NER")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--events-output",
        type=Path,
        default=None,
        help="Also write just the events list (no irrelevant_messages wrapper) here -- "
        "this is the shape Ronav's activity_baseline.py expects as input.",
    )
    args = parser.parse_args()

    output = run_pipeline(args.input, args.relevance_model, args.ner_model, args.embedding_model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    n_dupes = sum(1 for e in output["events"] if e["duplicate_of"])
    print(
        f"Wrote {len(output['events'])} events ({n_dupes} flagged as duplicates) "
        f"and {len(output['irrelevant_messages'])} filtered-out messages to {args.output}"
    )

    if args.events_output:
        args.events_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.events_output, "w", encoding="utf-8") as f:
            json.dump(output["events"], f, indent=2, ensure_ascii=False)
        print(f"Also wrote the bare events list to {args.events_output}")


if __name__ == "__main__":
    main()
