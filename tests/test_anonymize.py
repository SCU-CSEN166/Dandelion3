import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_phone_and_age_redacted_without_model():
    """Phone/age redaction is pure regex, so this doesn't need the NER model."""
    from anonymize import AGE_RE, PHONE_RE

    assert PHONE_RE.sub("[PHONE]", "call her at +254712345678 please") == "call her at [PHONE] please"
    assert AGE_RE.sub("[AGE]", "she was 67 years old today") == "she was [AGE] today"
    assert AGE_RE.sub("[AGE]", "patient John O. (age 54) is stable") == "patient John O. [AGE] is stable"


def test_person_names_fully_redacted():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from anonymize import anonymize_text

    result = anonymize_text("Visited Mama Achieng's homestead today, gave 2 goats.")
    assert "Achieng" not in result
    assert "[PERSON]" in result


def test_location_is_not_redacted():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from anonymize import anonymize_text

    result = anonymize_text("Conducted a nutrition training session at the Kakamega community hall.")
    assert "Kakamega" in result


def test_name_split_into_multiple_entities_does_not_corrupt_text():
    """Regression test: dslim/bert-base-NER sometimes tags one name as 2-3
    separate adjacent PER entities ("Otieno" -> "O" + "##tien" + "##o"),
    which used to produce corrupted output like "[PERSON]N]N]"."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from anonymize import anonymize_text

    result = anonymize_text("Checking on the Otieno household's new goat shed construction.")
    assert result == "Checking on the [PERSON] household's new goat shed construction."

    result = anonymize_text(
        "Visited the Oduya family, gave 2 goats as part of the livelihood program, they were thrilled."
    )
    assert result == "Visited the [PERSON] family, gave 2 goats as part of the livelihood program, they were thrilled."
