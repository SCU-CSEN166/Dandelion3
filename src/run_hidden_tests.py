"""Load supplied transcripts and run the existing models; no ASR or gold labels.

Run from the repository root:
    python src/run_hidden_tests.py --input hidden_test_transcripts.xlsx
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from anonymize import anonymize_text
from activity_review import process_messages
from cluster import cluster_events
from dedup import find_duplicates
from extract import extract_attributes
from relevance_classifier import classify_relevance


FIELDS = (
    "Message_id", "anonymized_text", "relevance", "activity_type",
    "participants", "duplicate_of",
)


def load_transcripts(path: Path, sheet_name: str | None = None) -> list[dict]:
    """Read rows in workbook order, preserving the supplied integer message IDs."""
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook[sheet_name] if sheet_name else workbook.worksheets[0]
        rows = sheet.iter_rows(values_only=True)
        header = next(rows, ())
        columns = {str(value).strip().lower(): i for i, value in enumerate(header)}
        required = {"message_id", "transcript"}
        if not required <= columns.keys():
            raise ValueError("Workbook needs message_id and transcript column headers in row 1")
        messages = []
        seen = set()
        for row_number, row in enumerate(rows, start=2):
            if all(value is None or value == "" for value in row):
                continue
            identifier = row[columns["message_id"]]
            if isinstance(identifier, bool) or identifier is None:
                raise ValueError(f"Row {row_number}: message_id must be an integer")
            try:
                message_id = int(identifier)
                if float(identifier) != message_id:
                    raise ValueError
            except (ValueError, TypeError, OverflowError):
                raise ValueError(f"Row {row_number}: message_id must be an integer") from None
            if message_id in seen:
                raise ValueError(f"Row {row_number}: duplicate message_id {message_id}")
            transcript = row[columns["transcript"]]
            if not isinstance(transcript, str) or not transcript.strip():
                raise ValueError(f"Row {row_number}: transcript must contain text")
            seen.add(message_id)
            messages.append({"id": message_id, "text": transcript.strip()})
        if not messages:
            raise ValueError("Workbook contains no transcripts")
        return messages
    finally:
        workbook.close()


def run_hidden_tests(
    messages: list[dict],
    relevance_model: str = "facebook/bart-large-mnli",
    ner_model: str = "dslim/bert-base-NER",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> list[dict]:
    # No sender/time metadata is supplied; clustering can still use text and counts.
    classified = classify_relevance([dict(message) for message in messages], relevance_model)
    if [item["id"] for item in classified] != [item["id"] for item in messages]:
        raise ValueError("Classifier did not return one ordered result per input message")
    relevant = []
    for message in classified:
        if message["predicted_label"] not in {"relevant", "irrelevant"}:
            raise ValueError(f"Unexpected relevance label: {message['predicted_label']}")
        if message["predicted_label"] == "relevant":
            relevant.append({
                **message,
                **extract_attributes(message["text"], ""),
                "relevance_confidence": message["confidence"],
            })

    events = cluster_events(relevant, embedding_model)
    order = {message["id"]: i for i, message in enumerate(messages)}
    events.sort(key=lambda event: min(order[mid] for mid in event["source_message_ids"]))
    for i, event in enumerate(events, start=1):
        event.update(event_id=f"evt_{i:04d}", relevance="event")
    find_duplicates(events, embedding_model)
    reviewed = process_messages(events, ner_model=ner_model)
    by_message = {mid: event for event in reviewed for mid in event["source_message_ids"]}
    first_messages = {
        event["event_id"]: min(event["source_message_ids"], key=order.get)
        for event in reviewed
    }

    results = []
    for message in classified:
        event = by_message.get(message["id"])
        attributes = event["attributes"] if event else {}
        duplicate_of = "none"
        if event:
            # Include reports merged by clustering as well as event-level duplicates.
            first_id = first_messages[event.get("duplicate_of") or event["event_id"]]
            if first_id != message["id"]:
                duplicate_of = first_id
        participants = attributes.get("participants")
        results.append({
            "Message_id": message["id"],
            "anonymized_text": anonymize_text(message["text"], ner_model),
            "relevance": "Event" if event else "Chit-chat",
            "activity_type": attributes.get("activity_type") or "",
            "participants": participants if participants is not None else "",
            "duplicate_of": duplicate_of,
        })
    return results


def write_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("hidden_test_transcripts.xlsx"))
    parser.add_argument("--sheet", help="Worksheet name (default: first worksheet)")
    parser.add_argument("--output", type=Path, default=Path("outputs/hidden_test_outputs.csv"))
    parser.add_argument("--check-input", action="store_true", help="Validate workbook without loading models")
    parser.add_argument("--relevance-model", default="facebook/bart-large-mnli")
    parser.add_argument("--ner-model", default="dslim/bert-base-NER")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    messages = load_transcripts(args.input, args.sheet)
    print(f"Loaded {len(messages)} transcripts from {args.input}")
    if args.check_input:
        return
    print("Using supplied transcripts; ASR is not evaluated.")
    results = run_hidden_tests(messages, args.relevance_model, args.ner_model, args.embedding_model)
    write_results(results, args.output)
    events = sum(row["relevance"] == "Event" for row in results)
    duplicates = sum(row["duplicate_of"] != "none" for row in results)
    print(f"Saved {len(results)} rows: {events} events, {len(results) - events} chit-chat, "
          f"{duplicates} duplicates -> {args.output}")
    print("Accuracy/F1 require gold labels; this file contains predictions only.")


if __name__ == "__main__":
    main()
