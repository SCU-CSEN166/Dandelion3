import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extract import extract_attributes, to_iso_date


def test_to_iso_date():
    assert to_iso_date("03/02/2026") == "2026-02-03"


def test_livelihood_distribution_with_quantity_and_outcome():
    attrs = extract_attributes(
        "Visited [PERSON]'s homestead today, gave 2 goats as part of the livelihood program. "
        "She was [AGE] and very happy with the support.",
        "03/02/2026",
    )
    assert attrs["activity_type"] == "livelihood_distribution"
    assert attrs["quantity"] == "2 goats"
    assert attrs["outcome"] == "positive"
    assert attrs["date"] == "2026-02-03"


def test_training_with_location():
    attrs = extract_attributes(
        "Conducted a nutrition training session for 14 mothers at the Kakamega community hall.",
        "03/02/2026",
    )
    assert attrs["activity_type"] == "training"
    assert attrs["location"] == "Kakamega"
    assert attrs["quantity"] == "14 mothers"


def test_unmatched_text_falls_back_to_other():
    attrs = extract_attributes("blessings to everyone, stay safe out there today", "03/02/2026")
    assert attrs["activity_type"] == "other"
    assert attrs["quantity"] is None
    assert attrs["location"] is None
