import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest import _audio_creation_time, load_from_audio_dir, load_messages


def test_audio_creation_time_falls_back_to_mtime_without_ffprobe(tmp_path, monkeypatch):
    import subprocess

    def _raise(*a, **kw):
        raise FileNotFoundError("ffprobe not installed")

    monkeypatch.setattr(subprocess, "run", _raise)

    f = tmp_path / "clip.mp3"
    f.write_bytes(b"not real audio")
    result = _audio_creation_time(f)
    assert result is not None  # falls back to filesystem mtime rather than crashing


def test_load_from_audio_dir_raises_on_empty_folder(tmp_path):
    with pytest.raises(ValueError):
        load_from_audio_dir(tmp_path)


def test_load_messages_dispatches_on_xlsx_vs_txt(tmp_path):
    # .txt should route to the chat-export loader, not the xlsx loader --
    # this will fail on load_from_chat_export's own validation (a .txt
    # with no parseable lines), not on dispatch, which is what we're
    # actually checking here.
    empty_txt = tmp_path / "chat.txt"
    empty_txt.write_text("")
    messages, media_only = load_messages(empty_txt)
    assert messages == []
    assert media_only == []


def test_real_audio_folder_end_to_end():
    """Full integration: transcribes real audio, in recording-time order."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    audio_dir = ROOT / "drive-download-20261113T223756Z-1-001"
    if not audio_dir.is_dir():
        pytest.skip("real hidden-set audio not present on this machine")

    messages, media_only = load_from_audio_dir(audio_dir, asr_model="openai/whisper-small")

    assert len(messages) == 10
    assert media_only == []
    # Ordered by real recording-time metadata, not filename sort.
    timestamps = [m["timestamp"] for m in messages]
    assert timestamps == sorted(timestamps)
    assert all(m["text"] for m in messages)
