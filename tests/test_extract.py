import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extract import extract_attributes, extract_participants, to_iso_date, words_to_number


def test_to_iso_date():
    assert to_iso_date("03/02/2026") == "2026-02-03"


def test_words_to_number():
    assert words_to_number("forty two") == 42
    assert words_to_number("twenty five") == 25
    assert words_to_number("eighteen thousand") == 18000
    assert words_to_number("two hundred") == 200
    assert words_to_number("not a number") is None


def test_participants_digit_form():
    assert extract_participants("42 women came to the session") == 42


def test_participants_word_form():
    # The real hidden-set data used almost entirely word-form numbers --
    # this is the case the old digit-only regex completely missed.
    assert extract_participants("Twenty five members attended out of thirty") == 25
    assert extract_participants("We screened twenty eight women today") == 28


def test_participants_none_when_absent():
    assert extract_participants("We had a good meeting today.") is None


def test_quantity_still_extracted_for_goods():
    attrs = extract_attributes("Gave out 2 goats as part of the livelihood program.", "03/02/2026")
    assert attrs["quantity"] == "2 goats"


def test_activity_type_is_free_text_not_forced_into_fixed_enum():
    attrs = extract_attributes(
        "Today we held a nutrition talk with 15 mothers.", "03/02/2026"
    )
    assert "nutrition talk" in attrs["activity_type"]
    assert attrs["participants"] == 15


def test_activity_type_skips_filler_words_after_verb():
    # Regression: "held this afternoon" used to capture "this afternoon"
    # as the activity name instead of falling through to the real phrase.
    attrs = extract_attributes(
        "The community dialogue on violence we held this afternoon at the camp.", "03/02/2026"
    )
    assert attrs["activity_type"] != "this afternoon"


def test_activity_type_does_not_trigger_on_common_idiom():
    # Regression: "ran out of stock" used to get captured as an activity
    # name because "ran" was in the verb trigger list.
    attrs = extract_attributes("We ran out of stock for the implants.", "03/02/2026")
    assert attrs["activity_type"] != "out of stock"
