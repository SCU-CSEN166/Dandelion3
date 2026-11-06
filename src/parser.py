"""Step 1 - Parse a WhatsApp chat export into ordered Message records.

Handles the two common export styles:
  Android: 2/3/26, 9:41 AM - Amina: Visited a homestead today
  iOS:     [2/3/26, 9:41:22 AM] Amina: Visited a homestead today
Multi-line messages, system lines, and audio attachments are supported.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

AUDIO_EXTS = (".opus", ".ogg", ".m4a", ".mp3", ".wav", ".aac", ".amr")

# Invisible characters WhatsApp inserts (LRM, narrow no-break space, etc.)
_INVISIBLE = dict.fromkeys(map(ord, "\u200e\u200f\u202a\u202b\u202c\u202d\u202e"), None)

# Android: "2/3/26, 9:41 AM - Sender: text"   iOS: "[2/3/26, 9:41:22 AM] Sender: text"
_LINE_RE = re.compile(
    r"^\[?(?P<date>\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]?"
    r"\s*(?:-\s*)?(?P<rest>.*)$"
)

# Attachment patterns
_ATTACH_PATTERNS = [
    re.compile(r"<attached:\s*(?P<f>[^>]+)>", re.I),            # iOS
    re.compile(r"^(?P<f>\S.*?\.\w{2,4})\s*\(file attached\)$", re.I),  # Android
    re.compile(r"^(?P<f>\S+\.\w{2,4})$"),                        # bare filename
]
_MEDIA_OMITTED = re.compile(r"<media omitted>|audio omitted", re.I)

_DATE_FORMATS = [
    "%m/%d/%y", "%m/%d/%Y", "%d/%m/%y", "%d/%m/%Y",
    "%Y-%m-%d", "%d.%m.%y", "%d.%m.%Y", "%m-%d-%y", "%d-%m-%Y",
]
_TIME_FORMATS = ["%I:%M %p", "%I:%M:%S %p", "%H:%M", "%H:%M:%S"]


@dataclass
class Message:
    msg_id: int
    timestamp: str  # ISO 8601
    sender: Optional[str]  # None for system messages
    text: str
    audio_file: Optional[str] = None  # filename as referenced in the chat
    audio_path: Optional[str] = None  # resolved path on disk (None if missing)
    is_voice_note: bool = False
    is_system: bool = False
    transcript: Optional[str] = None  # filled in by asr.py


def _clean(s: str) -> str:
    return s.translate(_INVISIBLE).replace("\u202f", " ").replace("\xa0", " ").strip()


def _parse_timestamp(date_s: str, time_s: str, dayfirst: Optional[bool]) -> datetime:
    date_s, time_s = date_s.strip(), time_s.strip().upper()
    date_fmts = list(_DATE_FORMATS)
    if dayfirst is True:
        date_fmts.sort(key=lambda f: f.startswith("%m"))
    for d in date_fmts:
        for t in _TIME_FORMATS:
            try:
                return datetime.strptime(f"{date_s} {time_s}", f"{d} {t}")
            except ValueError:
                continue
    raise ValueError(f"Unrecognized timestamp: {date_s} {time_s}")


def _detect_dayfirst(lines: list[str]) -> Optional[bool]:
    """If any date has a first field > 12 it's day-first; second field > 12 -> month-first."""
    for line in lines:
        m = _LINE_RE.match(_clean(line))
        if not m:
            continue
        parts = re.split(r"[/.\-]", m["date"])
        if len(parts) == 3 and len(parts[0]) <= 2:
            if int(parts[0]) > 12:
                return True
            if int(parts[1]) > 12:
                return False
    return None


def _find_attachment(text: str) -> Optional[str]:
    for pat in _ATTACH_PATTERNS:
        m = pat.search(text)
        if m:
            return m["f"].strip()
    return None


def parse_chat(
    chat_path: str | Path,
    audio_dir: Optional[str | Path] = None,
    dayfirst: Optional[bool] = None,
) -> list[Message]:
    """Parse a WhatsApp .txt export. Audio filenames are resolved against audio_dir
    (defaults to the folder containing the chat file).

    `dayfirst` overrides the auto-detection heuristic below, which can't tell
    "03/02/2026" is day-first when every date in the file has day and month
    both <= 12 (there's no ambiguous value to detect from) -- it silently
    falls back to month-first in that case. Pass True/False explicitly for
    exports you know the convention of (e.g. our synthetic data is day-first)."""
    chat_path = Path(chat_path)
    audio_dir = Path(audio_dir) if audio_dir else chat_path.parent
    lines = chat_path.read_text(encoding="utf-8-sig").splitlines()
    if dayfirst is None:
        dayfirst = _detect_dayfirst(lines)

    raw: list[dict] = []
    for line in lines:
        cleaned = _clean(line)
        m = _LINE_RE.match(cleaned)
        if m:
            try:
                ts = _parse_timestamp(m["date"], m["time"], dayfirst)
            except ValueError:
                m = None
            else:
                rest = m["rest"]
                if ": " in rest:
                    sender, text = rest.split(": ", 1)
                    raw.append({"ts": ts, "sender": sender.strip(), "text": text.strip(), "system": False})
                else:
                    raw.append({"ts": ts, "sender": None, "text": rest.strip(), "system": True})
                continue
        # continuation of previous message (multi-line)
        if raw and cleaned:
            raw[-1]["text"] += "\n" + cleaned

    messages: list[Message] = []
    for i, r in enumerate(raw):
        text = r["text"]
        audio_file = None if r["system"] else _find_attachment(text)
        is_voice = bool(audio_file and audio_file.lower().endswith(AUDIO_EXTS))
        if not is_voice:
            audio_file = None
        audio_path = None
        if is_voice:
            p = audio_dir / audio_file
            audio_path = str(p) if p.exists() else None
        messages.append(
            Message(
                msg_id=i,
                timestamp=r["ts"].isoformat(),
                sender=r["sender"],
                text="" if is_voice else text,
                audio_file=audio_file,
                audio_path=audio_path,
                is_voice_note=is_voice or bool(_MEDIA_OMITTED.search(text) and "audio" in text.lower()),
                is_system=r["system"],
            )
        )
    return messages


def save_messages(messages: list[Message], out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps([asdict(m) for m in messages], indent=2, ensure_ascii=False), encoding="utf-8")


def load_messages(path: str | Path) -> list[Message]:
    return [Message(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]


if __name__ == "__main__":
    import sys

    msgs = parse_chat(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    for m in msgs[:10]:
        print(m)
    print(f"\n{len(msgs)} messages, {sum(m.is_voice_note for m in msgs)} voice notes")
