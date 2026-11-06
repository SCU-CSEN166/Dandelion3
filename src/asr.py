"""Step 2 - Transcribe voice notes with Whisper (Hugging Face transformers).

Default model is openai/whisper-large-v3 (per project plan); use --model
openai/whisper-small if it is too slow / too big for your machine.
Requires ffmpeg on PATH to decode .opus/.ogg WhatsApp audio.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from parser import Message

DEFAULT_MODEL = "openai/whisper-large-v3"


class Transcriber:
    def __init__(self, model_name: str = DEFAULT_MODEL, language: Optional[str] = "en",
                 cache_dir: Optional[str | Path] = None):
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.language = language
        self.model_name = model_name
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=dtype,
            device=device,
            chunk_length_s=30,  # handles voice notes longer than 30s
        )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_file(self, audio_path: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        tag = self.model_name.replace("/", "_")
        return self.cache_dir / f"{Path(audio_path).stem}.{tag}.json"

    def transcribe(self, audio_path: str) -> str:
        cf = self._cache_file(audio_path)
        if cf and cf.exists():
            return json.loads(cf.read_text(encoding="utf-8"))["text"]
        kwargs = {"task": "transcribe"}
        if self.language:
            kwargs["language"] = self.language
        result = self.pipe(str(audio_path), generate_kwargs=kwargs)
        text = result["text"].strip()
        if cf:
            cf.write_text(json.dumps({"audio": str(audio_path), "text": text}, ensure_ascii=False), encoding="utf-8")
        return text


def transcribe_messages(messages: list[Message], transcriber: Transcriber) -> list[Message]:
    """Fill message.transcript for every voice note whose audio file exists."""
    for m in messages:
        if not m.is_voice_note:
            continue
        if not m.audio_path:
            print(f"[warn] msg {m.msg_id}: audio file missing ({m.audio_file}); skipping")
            continue
        try:
            m.transcript = transcriber.transcribe(m.audio_path)
        except Exception as e:  # keep going; one bad file shouldn't kill the run
            print(f"[error] msg {m.msg_id}: {e}")
    return messages
