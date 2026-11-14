"""
Build a safe-to-commit "new base test case" from the real hidden-set
transcripts (data/hidden_test_transcripts.json): swap every real-sounding
name and phone number for a clearly fake one, keeping everything else
(content, ASR quirks, structure) intact. Names are mapped consistently so
the same person gets the same fake name everywhere they're mentioned --
that matters for testing clustering/dedup, which relies on shared names
across messages about the same event.

Usage:
    python scripts/deidentify_hidden_set.py
Writes data/synthetic/hidden_set_deidentified.json and a companion gold
labels file (data/synthetic/hidden_set_gold.json), built from the manual
read-through of all 10 transcripts.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/hidden_test_transcripts.json"
OUT_MESSAGES = ROOT / "data/synthetic/hidden_set_deidentified.json"
OUT_GOLD = ROOT / "data/synthetic/hidden_set_gold.json"

# Exact substring replacements, applied in order. Longer/more specific
# strings first so partial names don't get replaced incorrectly.
NAME_REPLACEMENTS = [
    ("Grace Wengiru", "Nyokabi Kamau"),
    ("faith aching", "Zawadi Otieno"),
    ("Grace", "Nyokabi"),
    ("Rose in Jerry", "Winnie Chege"),
    ("the Beatrice Latino", "Consolata Mwangi"),
    ("Beatrice Latino", "Consolata Mwangi"),
    ("The Trejara Mama Lucy Key Prop", "The treasurer, Mama Esther Njoki,"),
    ("Lucy's number", "Esther's number"),
    ("Janet Mongi", "Dorcas Kiptoo"),
    ("Mrs. Esther Koch", "Mrs. Ann Wafula"),
    ("Masi Wambui", "Pauline Cherono"),
    ("Corporal and Mutua", "Corporal Jane Omondi"),
    ("Nancy Shia Onoh", "Teresia Mburu"),
    ("sister Nancy", "sister Teresia"),
    ("Sister Nancy", "Sister Teresia"),
    ("Mrs Agnes Wairimu", "Mrs. Beatrice Simiyu"),
    ("Mama Parede Jeb2", "Mama Clara Wekesa"),
    ("Ruth", "Florence Macharia"),
    ("Caroline Nambura", "Edith Towett"),
    ("Mr. Pita Lungat", "Mr. Samuel Kiprotich"),
    ("Dray Smootoni", "Lydia Nafula"),
]

PHONE_RE = re.compile(r"0\d{2,3}[\s.-]?\d{3,4}[\s.-]?\d{2,4}")


def deidentify(text: str, fake_number_counter: list[int]) -> str:
    for real, fake in NAME_REPLACEMENTS:
        text = text.replace(real, fake)

    def _replace_phone(match):
        fake_number_counter[0] += 1
        return f"0700-000-{100 + fake_number_counter[0]}"

    return PHONE_RE.sub(_replace_phone, text)


# Manually reviewed ground truth (from reading all 10 real transcripts).
GOLD = {
    "msg01": {"relevant": True, "event_group": "family_planning_outreach", "participants": 42},
    "msg02": {"relevant": True, "event_group": "family_planning_outreach", "participants": 42},
    "msg03": {"relevant": False, "event_group": None, "participants": None},
    "msg04": {"relevant": True, "event_group": "table_banking", "participants": 25},
    "msg05": {"relevant": True, "event_group": "menstrual_hygiene_session", "participants": 60},
    "msg06": {"relevant": True, "event_group": "gbv_dialogue", "participants": 35},
    "msg07": {"relevant": True, "event_group": "cervical_cancer_screening", "participants": 28},
    "msg08": {"relevant": False, "event_group": None, "participants": None},
    "msg09": {"relevant": True, "event_group": "kitchen_garden_training", "participants": 18},
    "msg10": {"relevant": True, "event_group": "cervical_cancer_screening", "participants": 28},
}


def main():
    rows = json.loads(SRC.read_text())
    counter = [0]
    deidentified = [
        {"message_id": r["message_id"], "text": deidentify(r["transcript"], counter)} for r in rows
    ]

    OUT_MESSAGES.parent.mkdir(parents=True, exist_ok=True)
    OUT_MESSAGES.write_text(json.dumps(deidentified, indent=2, ensure_ascii=False))
    OUT_GOLD.write_text(json.dumps(GOLD, indent=2, ensure_ascii=False))
    print(f"Wrote {len(deidentified)} de-identified messages to {OUT_MESSAGES}")
    print(f"Wrote gold labels to {OUT_GOLD}")


if __name__ == "__main__":
    main()
