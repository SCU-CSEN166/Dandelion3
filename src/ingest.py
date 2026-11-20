"""
Shared input loading for the full pipeline. One function, `load_messages`,
accepts any of:
  - a WhatsApp .txt export (with or without voice-note attachments)
  - a plain folder of raw audio/video files, with no chat export wrapping
    them -- ordered by each file's embedded creation-time metadata when
    available (via ffprobe), falling back to filesystem modification time
  - a single audio/video file
  - an .xlsx of pre-supplied transcripts (Sathvik/Ronav's existing format)

and returns a uniform `(messages, media_only_messages)` pair: a list of
`{id, timestamp, sender, text}` dicts (timestamp/sender are `None` when
not knowable from the input) ready to feed into the rest of the pipeline.
"""

from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

from parser import AUDIO_EXTS, Message, parse_chat


def _message_text(m: Message) -> str:
    return m.transcript if (m.is_voice_note and m.transcript) else m.text


def _audio_creation_time(path: Path) -> str | None:
    """The file's embedded creation-time tag (via ffprobe) if present,
    else the filesystem's modification time -- either way, a real,
    orderable timestamp rather than just filename sort order."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format_tags=creation_time",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        value = result.stdout.strip()
        if value:
            return value.rstrip("Z")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    try:
        return datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    except OSError:
        return None


def load_from_chat_export(
    path: Path,
    audio_dir: Path | None = None,
    asr_model: str = "openai/whisper-small",
    dayfirst: bool = True,
) -> tuple[list[dict], list[dict]]:
    from clean import clean_message_text, is_media_only

    raw_messages = parse_chat(path, audio_dir=audio_dir, dayfirst=dayfirst)

    to_transcribe = [m for m in raw_messages if m.is_voice_note and m.transcript is None and m.audio_path]
    if to_transcribe:
        from asr import Transcriber, transcribe_messages

        print(f"Transcribing {len(to_transcribe)} voice note(s) with {asr_model}...")
        transcribe_messages(to_transcribe, Transcriber(asr_model))

    messages, media_only = [], []
    for m in raw_messages:
        if m.is_system:
            continue
        raw_text = _message_text(m)
        cleaned_text = clean_message_text(raw_text)
        if is_media_only(raw_text, cleaned_text):
            media_only.append({"id": m.msg_id, "sender": m.sender, "timestamp": m.timestamp})
            continue
        messages.append({"id": m.msg_id, "timestamp": m.timestamp, "sender": m.sender, "text": cleaned_text})
    return messages, media_only


def load_from_audio_dir(path: Path, asr_model: str = "openai/whisper-small") -> tuple[list[dict], list[dict]]:
    from asr import Transcriber

    files = sorted(
        (f for f in path.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXTS),
        key=lambda f: (_audio_creation_time(f) or "", f.name),
    )
    if not files:
        raise ValueError(f"No audio files found in {path} (looked for {AUDIO_EXTS})")

    transcriber = Transcriber(asr_model)
    print(f"Transcribing {len(files)} audio file(s) with {asr_model}, ordered by recording time...")
    messages = []
    for i, f in enumerate(files):
        text = transcriber.transcribe(str(f))
        messages.append({"id": i, "timestamp": _audio_creation_time(f), "sender": None, "text": text})
    return messages, []


def load_from_xlsx(path: Path, sheet_name: str | None = None) -> tuple[list[dict], list[dict]]:
    from run_hidden_tests import load_transcripts

    messages = load_transcripts(path, sheet_name)
    for m in messages:
        m.setdefault("timestamp", None)
        m.setdefault("sender", None)
    return messages, []


def load_messages(
    path: Path,
    asr_model: str = "openai/whisper-small",
    audio_dir: Path | None = None,
    dayfirst: bool = True,
    sheet_name: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Dispatches on what kind of input `path` is. See module docstring."""
    path = Path(path)
    if path.is_dir():
        return load_from_audio_dir(path, asr_model)
    if path.suffix.lower() == ".xlsx":
        return load_from_xlsx(path, sheet_name)
    if path.suffix.lower() in AUDIO_EXTS:
        from asr import Transcriber

        print(f"Transcribing 1 audio file with {asr_model}...")
        text = Transcriber(asr_model).transcribe(str(path))
        return [{"id": 0, "timestamp": _audio_creation_time(path), "sender": None, "text": text}], []
    return load_from_chat_export(path, audio_dir=audio_dir, asr_model=asr_model, dayfirst=dayfirst)
