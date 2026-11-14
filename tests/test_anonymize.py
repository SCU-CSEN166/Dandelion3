"""
anonymize.py isn't called by our own pipeline.py anymore (anonymization
moved to Ronav's side), but we still own and maintain this module for him,
so it keeps its own tests.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_phone_and_age_redacted_without_model():
    from anonymize import AGE_RE, PHONE_RE

    assert PHONE_RE.sub("[PHONE]", "call her at +254712345678 please") == "call her at [PHONE] please"
    assert AGE_RE.sub("[AGE]", "she was 67 years old today") == "she was [AGE] today"


def test_person_names_fully_redacted():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from anonymize import anonymize_text

    result = anonymize_text("Visited Mama Achieng's homestead today, gave 2 goats.")
    assert "Achieng" not in result
    assert "[PERSON]" in result


def test_locations_are_now_redacted():
    # Team decision reversed the earlier "don't redact locations" choice,
    # to match the golden/solution set's expected behavior.
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from anonymize import anonymize_text

    result = anonymize_text("Six women were referred to Rongai Health Centre for implants.")
    assert "Rongai" not in result
    assert "[LOCATION]" in result


def test_name_split_into_multiple_entities_does_not_corrupt_text():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from anonymize import anonymize_text

    result = anonymize_text("Checking on the Otieno household's new goat shed construction.")
    assert result == "Checking on the [PERSON] household's new goat shed construction."
