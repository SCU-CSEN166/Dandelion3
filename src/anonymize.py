"""
Anonymization (pipeline step 3): redact PII from message text before it goes
any further in the pipeline, using a BERT NER model for person names plus
regexes for phone numbers and ages.

Locations are intentionally NOT redacted here: Dandelion Africa's M&E
reporting needs a location (village/area) per activity, and the place names
in our synthetic data are program sites, not personal addresses. This is a
real privacy trade-off, not an oversight — see README "Known limitations".
"""

import re
from functools import lru_cache

PHONE_RE = re.compile(r"\+?\d[\d\-\s]{7,}\d")
AGE_RE = re.compile(r"\(?\s*age\s*\d{1,3}\s*\)?|\b\d{1,3}\s*years?\s*old\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _get_ner_pipeline(model_name: str):
    from transformers import pipeline

    return pipeline("ner", model=model_name, aggregation_strategy="simple")


def _extend_to_word_boundary(text: str, start: int, end: int) -> tuple[int, int]:
    """BERT wordpiece tokenization sometimes tags only part of an unfamiliar
    name (e.g. "Lilian" -> just "Lil"). Extend the span to the full
    contiguous word on each side, since under-redacting a name is the
    worse failure mode for anonymization."""
    while start > 0 and text[start - 1].isalpha():
        start -= 1
    while end < len(text) and text[end].isalpha():
        end += 1
    return start, end


def _merge_overlapping_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """dslim/bert-base-NER sometimes returns a single name as 2-3 separate
    adjacent PER entities (e.g. "Otieno" -> "O" + "##tien" + "##o"), which
    after word-boundary extension become overlapping/duplicate spans.
    Replacing those independently corrupts the text (stale offsets after
    the first replacement), so merge overlapping spans first."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for start, end in spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def anonymize_text(text: str, model_name: str = "dslim/bert-base-NER") -> str:
    ner = _get_ner_pipeline(model_name)
    entities = ner(text)

    spans = [
        _extend_to_word_boundary(text, e["start"], e["end"])
        for e in entities
        if e["entity_group"] == "PER"
    ]
    spans = _merge_overlapping_spans(spans)

    redacted = text
    for start, end in sorted(spans, reverse=True):
        redacted = redacted[:start] + "[PERSON]" + redacted[end:]

    redacted = PHONE_RE.sub("[PHONE]", redacted)
    redacted = AGE_RE.sub("[AGE]", redacted)
    return redacted
