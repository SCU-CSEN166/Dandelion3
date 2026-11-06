"""
Full mid-pipeline baseline covering Roland's scope: anonymize -> relevance
filter -> event segmentation -> attribute extraction -> deduplication.

Produces the shared JSON handoff format for the rest of the team (this is
what Ronav's formatting/testing step consumes).

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
from relevance_classifier import classify_relevance, parse_whatsapp_export
from segment import segment_events


def run_pipeline(
    input_path: Path,
    relevance_model: str = "facebook/bart-large-mnli",
    ner_model: str = "dslim/bert-base-NER",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict:
    messages = parse_whatsapp_export(input_path)
    print(f"Parsed {len(messages)} messages from {input_path}")

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
        first_message = messages_by_id[event["source_message_ids"][0]]
        event["attributes"] = extract_attributes(event["anonymized_text"], first_message["date"])
        event["relevance"] = {
            "label": "relevant",
            "confidence": max(
                messages_by_id[mid]["relevance_confidence"] for mid in event["source_message_ids"]
            ),
        }
        del event["sender"]

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


if __name__ == "__main__":
    main()
