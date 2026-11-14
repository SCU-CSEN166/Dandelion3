import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clean import clean_message_text, is_media_only


def test_strips_media_omitted_placeholder():
    assert clean_message_text("<Media omitted>") == ""
    assert clean_message_text("image omitted") == ""
    assert clean_message_text("Check this out! video omitted") == "Check this out!"


def test_strips_deleted_message_placeholder():
    assert clean_message_text("This message was deleted") == ""
    assert clean_message_text("You deleted this message") == ""


def test_real_text_untouched():
    text = "Visited the homestead today, gave 2 goats."
    assert clean_message_text(text) == text


def test_is_media_only():
    raw = "<Media omitted>"
    assert is_media_only(raw, clean_message_text(raw))
    raw2 = "Visited the homestead today."
    assert not is_media_only(raw2, clean_message_text(raw2))


def test_empty_message_is_not_media_only():
    # An already-empty message isn't a media placeholder being stripped --
    # don't misreport it as one.
    assert not is_media_only("", clean_message_text(""))
