import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parser import parse_chat  # noqa: E402

ANDROID = """2/3/26, 9:41 AM - Messages are end-to-end encrypted.
2/3/26, 9:41 AM - Amina: Visited a homestead today
gave 2 goats to the family
2/3/26, 9:45 AM - Wanjiru: PTT-20260203-WA0012.opus (file attached)
2/3/26, 9:50 AM - Amina: lol see you tomorrow
"""

IOS = """[2/3/26, 9:41:22 AM] Amina: Training done
[2/3/26, 9:42:10 AM] Wanjiru: \u200e<attached: 00000012-AUDIO-2026-02-03.opus>
"""


def test_android(tmp_path):
    (tmp_path / "chat.txt").write_text(ANDROID)
    (tmp_path / "PTT-20260203-WA0012.opus").write_bytes(b"x")
    msgs = parse_chat(tmp_path / "chat.txt")
    assert len(msgs) == 4
    assert msgs[0].is_system
    assert msgs[1].text == "Visited a homestead today\ngave 2 goats to the family"
    assert msgs[2].is_voice_note and msgs[2].audio_path
    assert msgs[3].timestamp.startswith("2026-02-03T09:50")


def test_ios_missing_audio(tmp_path):
    (tmp_path / "chat.txt").write_text(IOS)
    msgs = parse_chat(tmp_path / "chat.txt")
    assert msgs[1].is_voice_note and msgs[1].audio_path is None


def test_dayfirst_override():
    """Regression test: when every date in the file has day and month both
    <= 12 (e.g. "03/02/2026"), there's no ambiguous value for the
    auto-detect heuristic to find, so it silently falls back to
    month-first. Found while integrating with Roland's pipeline, which
    produced day-first synthetic data -- dates were coming out a month off.
    The explicit `dayfirst` override fixes this without touching the
    default auto-detect behavior (test_android/test_ios above still pass)."""
    content = "03/02/2026, 08:41 - Amina: Visited a homestead today\n"
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        chat_file = _Path(tmp) / "chat.txt"
        chat_file.write_text(content)

        auto = parse_chat(chat_file)
        assert auto[0].timestamp.startswith("2026-03-02"), "ambiguous date defaults to month-first"

        forced = parse_chat(chat_file, dayfirst=True)
        assert forced[0].timestamp.startswith("2026-02-03"), "dayfirst=True should read it as 3 Feb"
