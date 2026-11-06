"""
Attribute extraction (pipeline step 6) -- rule/keyword baseline.

The Week 5 plan proposes a schema-constrained Llama prompt for this step.
We're starting with a keyword/regex baseline instead (no local Llama setup
yet — see README "Known limitations"), matching the Week 5 "simplest
baseline" philosophy. Swap in the LLM version once Ollama is set up.
"""

from __future__ import annotations

import re

ACTIVITY_RULES = [
    ("livelihood_distribution", [r"\bgoats?\b", r"\bseed packets?\b", r"\bchickens?\b"]),
    ("training", [r"\btraining\b", r"\bhygiene and handwashing\b"]),
    ("health_visit", [r"\bprenatal\b", r"\bcheckup\b", r"\bpatient\b", r"\bmedication\b"]),
    ("follow_up", [r"\bfollow(ed)?[- ]up\b", r"\bfollow-up call\b"]),
    ("distribution", [r"\bdistributed\b", r"\bmosquito nets?\b", r"\bfood packages?\b", r"\bgave out\b"]),
    ("intake_assessment", [r"\bintake\b", r"\breferred\b"]),
]

QUANTITY_RE = re.compile(r"\b(\d+)\s+([a-zA-Z][\w\- ]{1,25}?)\b(?=[.,]| as| to| in| for| at| from|$)")

KNOWN_LOCATIONS = ["Kakamega", "Shibuye"]
LOCATION_RE = re.compile(r"\bin the ([A-Z][a-zA-Z]+) area\b")

OUTCOME_KEYWORDS = {
    "positive": ["happy", "delighted", "pleased", "positive", "progressing normally", "doing well", "sold eggs"],
    "needs_followup": ["no answer", "will try again", "next visit scheduled"],
}


def extract_activity_type(text: str) -> str:
    lowered = text.lower()
    for label, patterns in ACTIVITY_RULES:
        if any(re.search(pattern, lowered) for pattern in patterns):
            return label
    return "other"


def extract_quantity(text: str) -> str | None:
    match = QUANTITY_RE.search(text)
    if match:
        return f"{match.group(1)} {match.group(2)}".strip()
    return None


def extract_location(text: str) -> str | None:
    for location in KNOWN_LOCATIONS:
        if location in text:
            return location
    match = LOCATION_RE.search(text)
    return match.group(1) if match else None


def extract_outcome(text: str) -> str:
    lowered = text.lower()
    for label, keywords in OUTCOME_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return label
    return "unknown"


def to_iso_date(date_str: str) -> str:
    day, month, year = date_str.split("/")
    return f"{year}-{month}-{day}"


def extract_attributes(anonymized_text: str, source_date: str) -> dict:
    return {
        "activity_type": extract_activity_type(anonymized_text),
        "date": to_iso_date(source_date),
        "location": extract_location(anonymized_text),
        "quantity": extract_quantity(anonymized_text),
        "outcome": extract_outcome(anonymized_text),
    }
