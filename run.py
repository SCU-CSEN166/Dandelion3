#!/usr/bin/env python3
"""
One-command entry point for the whole pipeline.

Give it any of:
  - a WhatsApp .txt export (voice notes get transcribed automatically)
  - a folder of raw voice-note audio/video files (ordered by each file's
    recording-time metadata when available, transcribed with Whisper)
  - a single audio/video file
  - an .xlsx of pre-supplied transcripts

and it runs ingestion -> ASR (if needed) -> media cleanup -> relevance
filtering -> attribute extraction -> event clustering -> deduplication ->
anonymization -> review flagging, and writes one output CSV.

Usage:
    python run.py <input_path> [--output hidden_test_output_post_changes.csv] [--asr-model MODEL]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ingest import load_messages
from run_hidden_tests import run_hidden_tests, write_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="WhatsApp .txt export, folder of audio files, a single audio file, or an .xlsx of transcripts")
    parser.add_argument("--output", type=Path, default=Path("hidden_test_output_post_changes.csv"))
    parser.add_argument("--asr-model", default="openai/whisper-small")
    parser.add_argument("--relevance-model", default="facebook/bart-large-mnli")
    parser.add_argument("--ner-model", default="dslim/bert-base-NER")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    messages, media_only = load_messages(args.input, asr_model=args.asr_model)
    extra = f" ({len(media_only)} media-only set aside)" if media_only else ""
    print(f"Loaded {len(messages)} text message(s){extra} from {args.input}")

    if not messages:
        print("Nothing to process -- no text messages after ingestion.")
        return

    results = run_hidden_tests(messages, args.relevance_model, args.ner_model, args.embedding_model)
    write_results(results, args.output)

    n_events = sum(row["relevance"] == "Event" for row in results)
    n_duplicates = sum(row["duplicate_of"] != "none" for row in results)
    print(
        f"Saved {len(results)} rows: {n_events} events, {len(results) - n_events} chit-chat, "
        f"{n_duplicates} duplicates -> {args.output}"
    )


if __name__ == "__main__":
    main()
