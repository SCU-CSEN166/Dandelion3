"""
Full mid-pipeline covering Roland's scope: relevance filter -> per-message
extraction -> whole-chat event clustering -> duplicate-event safety net.

Architecture change from the original version: anonymization has moved to
Ronav's side. We hand off RAW (unredacted) text; he anonymizes it as the
last step before his review layer. Classification, extraction, and
clustering all run on raw text here -- see cluster.py's docstring for why
that's not just "doing it in a different order," but actually fixes real
dedup failures.

Also new: a step to strip WhatsApp media-placeholder text ("<Media
omitted>", etc.) before anything else sees it, and a dedicated
`participants` field (separate from `quantity`, which is for goods/items).

Usage:
    python src/pipeline.py \
        --input data/synthetic/whatsapp_export.txt \
        --output outputs/pipeline_output.json
"""

import argparse
import json
from pathlib import Path

from clean import clean_message_text, is_media_only
from cluster import cluster_events
from dedup import find_duplicates
from extract import extract_attributes
from parser import Message, parse_chat
from relevance_classifier import classify_relevance


def _message_text(m: Message) -> str:
    """Prefer the ASR transcript for voice notes; fall back to the raw text."""
    return m.transcript if (m.is_voice_note and m.transcript) else m.text


def run_pipeline(
    input_path: Path,
    relevance_model: str = "facebook/bart-large-mnli",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    asr_model: str = "openai/whisper-small",
    cluster_threshold: float = 0.5,
    dedup_threshold: float = 0.75,
    dayfirst: bool = True,
    audio_dir=None,
) -> dict:
    raw_messages = parse_chat(input_path, audio_dir=audio_dir, dayfirst=dayfirst)

    # Step: ASR -- transcribe any voice notes that don't already have a transcript.
    to_transcribe = [m for m in raw_messages if m.is_voice_note and m.transcript is None and m.audio_path]
    if to_transcribe:
        from asr import Transcriber, transcribe_messages

        print(f"Transcribing {len(to_transcribe)} voice note(s) with {asr_model}...")
        transcribe_messages(to_transcribe, Transcriber(asr_model))

    # Step: clean media-placeholder text; split off anything that was ONLY a
    # media placeholder (no real text to classify).
    messages = []
    media_only_messages = []
    for m in raw_messages:
        if m.is_system:
            continue
        raw_text = _message_text(m)
        cleaned_text = clean_message_text(raw_text)
        if is_media_only(raw_text, cleaned_text):
            media_only_messages.append({"id": m.msg_id, "sender": m.sender, "timestamp": m.timestamp})
            continue
        messages.append({"id": m.msg_id, "timestamp": m.timestamp, "sender": m.sender, "text": cleaned_text})
    print(f"Parsed {len(messages)} text messages from {input_path} "
          f"({len(media_only_messages)} media-only messages set aside)")

    # Step: relevance filter on raw text (no anonymization before this).
    classified = classify_relevance(messages, relevance_model)
    for msg, result in zip(messages, classified):
        msg["predicted_label"] = result["predicted_label"]
        msg["relevance_confidence"] = result["confidence"]

    relevant_messages = [m for m in messages if m["predicted_label"] == "relevant"]
    irrelevant_messages = [m for m in messages if m["predicted_label"] == "irrelevant"]

    # Step: per-message extraction (activity_type, participants, etc.) --
    # happens BEFORE clustering, so clustering can use these as signals.
    for msg in relevant_messages:
        msg.update(extract_attributes(msg["text"], msg["timestamp"].split("T")[0]))

    # Step: cluster relevant messages into events across the WHOLE chat
    # (not just adjacent same-sender messages).
    events = cluster_events(relevant_messages, embedding_model, threshold=cluster_threshold)
    for event in events:
        event["event_id"] = None  # placeholder; assigned below after sorting
    events.sort(key=lambda e: e["source_message_ids"][0])
    for i, event in enumerate(events, start=1):
        event["event_id"] = f"evt_{i:04d}"
        event["relevance"] = "event"
        event["duplicate_of"] = None
        event["duplicate_confidence"] = None

    # Step: duplicate-event safety net (catches two separate clusters that
    # still turn out to be the same real event).
    find_duplicates(events, embedding_model, threshold=dedup_threshold)

    return {
        "events": events,
        "irrelevant_messages": [
            {
                "message_id": m["id"],
                "text": m["text"],
                "relevance": "chitchat",
                "confidence": m["relevance_confidence"],
            }
            for m in irrelevant_messages
        ],
        "media_only_messages": media_only_messages,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the Roland-scope mid-pipeline end-to-end.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--relevance-model", default="facebook/bart-large-mnli")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--asr-model", default="openai/whisper-small", help="Used only if the input has voice notes without an existing transcript.")
    parser.add_argument("--audio-dir", type=Path, default=None, help="Folder with audio attachments (default: the input file's folder).")
    parser.add_argument(
        "--events-output",
        type=Path,
        default=None,
        help="Also write just the events list (no irrelevant_messages wrapper) here -- "
        "this is the shape Ronav's activity_baseline.py expects as input.",
    )
    args = parser.parse_args()

    output = run_pipeline(
        args.input, args.relevance_model, args.embedding_model,
        args.asr_model, audio_dir=args.audio_dir,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    n_dupes = sum(1 for e in output["events"] if e["duplicate_of"])
    print(
        f"Wrote {len(output['events'])} events ({n_dupes} flagged as duplicates), "
        f"{len(output['irrelevant_messages'])} filtered-out messages, and "
        f"{len(output['media_only_messages'])} media-only messages to {args.output}"
    )

    if args.events_output:
        args.events_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.events_output, "w", encoding="utf-8") as f:
            json.dump(output["events"], f, indent=2, ensure_ascii=False)
        print(f"Also wrote the bare events list to {args.events_output}")


if __name__ == "__main__":
    main()
