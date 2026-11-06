import json
from datetime import datetime
from pathlib import Path
from parser import AUDIO_EXTS, Message

def _iso(ts):
    for f in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts.strip(), f).isoformat()
        except ValueError:
            pass
    return ts

def load_json_chat(path, audio_dir=None):
    path = Path(path)
    audio_dir = Path(audio_dir) if audio_dir else path.parent
    messages = []
    for i, it in enumerate(json.loads(path.read_text(encoding="utf-8-sig"))):
        audio_file = it.get("audio_file") or it.get("audio") or it.get("file")
        is_voice = str(it.get("type", "text")).lower() in ("voice", "audio", "voice_note", "ptt") or bool(
            audio_file and str(audio_file).lower().endswith(AUDIO_EXTS))
        audio_path = None
        if is_voice and audio_file and (audio_dir / audio_file).exists():
            audio_path = str(audio_dir / audio_file)
        messages.append(Message(
            msg_id=i, timestamp=_iso(str(it.get("timestamp", ""))), sender=it.get("sender"),
            text="" if is_voice else str(it.get("message", it.get("text", ""))),
            audio_file=audio_file if is_voice else None, audio_path=audio_path,
            is_voice_note=is_voice, transcript=it.get("transcript")))
    return messages
