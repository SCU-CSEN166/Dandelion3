"""
Attribute extraction -- rule/regex baseline.

Two notable changes from the original version:
- `activity_type` is now free text extracted directly from the message
  (e.g. "nutrition talk"), not a forced fit into a small fixed enum. The
  fixed-keyword categories turned out to badly overfit our own synthetic
  vocabulary (goats, mosquito nets) and miscategorized 6/7 real hidden-set
  events, often triggered by an incidental word ("follow-up," "referred")
  rather than the message's actual topic.
- `participants` is a dedicated numeric headcount field (distinct from
  `quantity`, which is for goods/items), and understands word-form numbers
  ("twenty five", "forty-two") as well as digits -- the hidden-set data
  used almost entirely word-form numbers, and the old digit-only regex
  missed every one of them.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Word-form numbers ("twenty five", "eighteen thousand")
# ---------------------------------------------------------------------------

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000}

_NUMBER_WORD = "|".join(list(_UNITS) + list(_TENS) + list(_SCALES))
NUMBER_PHRASE_RE = re.compile(rf"\b(?:{_NUMBER_WORD})(?:[\s-]+(?:{_NUMBER_WORD}))*\b", re.IGNORECASE)


def words_to_number(phrase: str) -> int | None:
    tokens = re.findall(r"[a-zA-Z]+", phrase.lower())
    total = 0
    current = 0
    found = False
    for tok in tokens:
        if tok in _UNITS:
            current += _UNITS[tok]
            found = True
        elif tok in _TENS:
            current += _TENS[tok]
            found = True
        elif tok in _SCALES:
            scale = _SCALES[tok]
            current = (current or 1) * scale
            if scale >= 1000:
                total += current
                current = 0
            found = True
    total += current
    return total if found else None


def _parse_count(text: str) -> int | None:
    text = text.strip()
    if text.isdigit():
        return int(text)
    return words_to_number(text)


# ---------------------------------------------------------------------------
# Participants (headcount) vs. quantity (goods/items)
# ---------------------------------------------------------------------------

PARTICIPANT_NOUNS = (
    r"mothers|women|girls|boys|men|members|farmers|children|people|attendees|"
    r"villagers|elders|patients|households|families|participants|students|pupils"
)
NUMBER_TOKEN = rf"(?:\d+|(?:{_NUMBER_WORD})(?:[\s-]+(?:{_NUMBER_WORD}))*)"
PARTICIPANTS_RE = re.compile(
    rf"\b({NUMBER_TOKEN})\s+({PARTICIPANT_NOUNS})\b", re.IGNORECASE
)

QUANTITY_RE = re.compile(r"\b(\d+)\s+([a-zA-Z][\w\- ]{1,25}?)\b(?=[.,]| as| to| in| for| at| from|$)")

KNOWN_LOCATIONS = ["Kakamega", "Shibuye"]
LOCATION_RE = re.compile(r"\bin the ([A-Z][a-zA-Z]+) area\b")

OUTCOME_KEYWORDS = {
    "positive": ["happy", "delighted", "pleased", "positive", "progressing normally", "doing well", "sold eggs", "normal"],
    "needs_followup": ["no answer", "will try again", "next visit scheduled", "follow-up", "follow up"],
    "negative": ["refused", "declined", "unable to", "could not", "did not attend"],
}

# Fixed categories kept only as a last-resort fallback -- see module docstring
# for why free-text extraction is tried first.
ACTIVITY_RULES = [
    ("livelihood_distribution", [r"\bgoats?\b", r"\bseed packets?\b", r"\bchickens?\b"]),
    ("training", [r"\btraining\b", r"\bhygiene and handwashing\b"]),
    ("health_visit", [r"\bprenatal\b", r"\bcheckup\b", r"\bpatient\b", r"\bmedication\b"]),
    ("distribution", [r"\bdistributed\b", r"\bmosquito nets?\b", r"\bfood packages?\b", r"\bgave out\b"]),
]

ACTIVITY_VERB_RE = re.compile(
    r"\b(?:held|conducted|did|organized|organised)\s+(?:the\s+|a\s+|an\s+)?"
    r"([a-zA-Z][\w\-' ]{2,60}?)"
    r"(?=\s+(?:at|in|with|for|today|this|yesterday|on)\b|[.,]|$)",
    re.IGNORECASE,
)
ACTIVITY_NOUN_RE = re.compile(
    r"\b(?:the\s+)?([a-zA-Z][\w\-' ]{2,60}?\s+"
    r"(?:session|training|talk|outreach|screening|dialogue|meeting|camp|programme|program|day|visit))\b",
    re.IGNORECASE,
)
_REPORTING_FROM_RE = re.compile(r"\breporting\s+from\b", re.IGNORECASE)

# A verb-phrase match starting with one of these is filler, not the real
# activity name (e.g. "held this afternoon" with the real noun phrase
# appearing *before* the verb instead) -- fall through to ACTIVITY_NOUN_RE.
_FILLER_FIRST_WORDS = {"this", "today", "yesterday", "that", "afternoon", "morning", "evening"}

# Self-referential/reporting lead-ins that make the extracted phrase read
# like a sentence fragment instead of a category (e.g. "i want to report
# on today's family planning outreach" -> "family planning outreach").
# Applied repeatedly since more than one can stack up.
_LEADIN_RES = [
    re.compile(r"^i\s+want\s+to\s+(?:report\s+on|share\s+about|talk\s+about)\s+", re.IGNORECASE),
    re.compile(r"^i\s+was\s+(?:also\s+)?at\s+", re.IGNORECASE),
    re.compile(r"^i\s+was\s+helping\s+[a-z]+(?:\s+[a-z]+)?\s+at\s+", re.IGNORECASE),
    re.compile(r"^this\s+is\s+[a-z]+(?:\s+[a-z]+)?\s+reporting\s+from\s+", re.IGNORECASE),
    re.compile(r"^we\s+went\s+to\s+[\w' ]+?\s+for\s+", re.IGNORECASE),
    re.compile(r"^(?:today|yesterday)\s+(?:is|was)\s+(?:our|the|an?)\s+", re.IGNORECASE),
    re.compile(r"^(?:today'?s|the)\s+", re.IGNORECASE),
]


def _strip_leadin(phrase: str) -> str:
    """Strip reporting/self-referential lead-in clauses so the result
    reads as a category, not a sentence fragment. Never returns an empty
    or too-short result -- falls back to the original phrase if a strip
    would leave nothing useful."""
    stripped = phrase
    changed = True
    while changed:
        changed = False
        for pattern in _LEADIN_RES:
            new_stripped = pattern.sub("", stripped, count=1)
            if new_stripped != stripped:
                stripped = new_stripped
                changed = True
    stripped = stripped.strip()
    return stripped if len(stripped) >= 3 else phrase


def extract_activity_type(text: str) -> str:
    match = ACTIVITY_VERB_RE.search(text)
    if match:
        phrase = match.group(1).strip()
        first_word = phrase.split()[0].lower() if phrase.split() else ""
        if first_word not in _FILLER_FIRST_WORDS:
            return _strip_leadin(phrase.lower())

    phrases = [match.group(1).strip().lower() for match in ACTIVITY_NOUN_RE.finditer(text)]
    if phrases:
        # An introduction may name the group or venue before the report names the activity.
        activity = next(
            (phrase for phrase in phrases if not _REPORTING_FROM_RE.search(phrase)),
            phrases[0],
        )
        return _strip_leadin(activity)

    lowered = text.lower()
    for label, patterns in ACTIVITY_RULES:
        if any(re.search(pattern, lowered) for pattern in patterns):
            return label

    return "other"


def extract_participants(text: str) -> int | None:
    match = PARTICIPANTS_RE.search(text)
    if not match:
        return None
    return _parse_count(match.group(1))


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


def extract_attributes(text: str, source_date: str) -> dict:
    """`source_date` accepts either `DD/MM/YYYY` or an already-ISO `YYYY-MM-DD`."""
    date_iso = to_iso_date(source_date) if "/" in source_date else source_date
    return {
        "activity_type": extract_activity_type(text),
        "date": date_iso,
        "location": extract_location(text),
        "participants": extract_participants(text),
        "quantity": extract_quantity(text),
        "outcome": extract_outcome(text),
    }
