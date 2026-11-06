"""Run steps 1-2: parse a WhatsApp export, then transcribe its voice notes.

Example:
    python src/run_parse_asr.py data/synthetic/chat.txt --audio-dir data/synthetic \
        --out output/messages.json --model openai/whisper-small
Use --no-asr to only parse.
"""
import argparse

from parser import parse_chat, save_messages
from json_loader import load_json_chat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chat", help="WhatsApp exported .txt file")
    ap.add_argument("--audio-dir", default=None, help="folder with audio attachments (default: chat's folder)")
    ap.add_argument("--out", default="output/messages.json")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--language", default="en")
    ap.add_argument("--no-asr", action="store_true")
    args = ap.parse_args()

    if args.chat.lower().endswith('.json'):
        messages = load_json_chat(args.chat, args.audio_dir)
    else:
        messages = parse_chat(args.chat, args.audio_dir)
    n_voice = sum(m.is_voice_note for m in messages)
    print(f"Parsed {len(messages)} messages ({n_voice} voice notes)")

    if not args.no_asr and n_voice:
        from asr import Transcriber, transcribe_messages

        t = Transcriber(args.model, language=args.language, cache_dir="output/asr_cache")
        transcribe_messages(messages, t)

    save_messages(messages, args.out)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
