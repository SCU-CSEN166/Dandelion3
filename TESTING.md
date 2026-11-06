# Testing reference (Roland's scope)

This covers the pipeline steps in `src/`: anonymization, relevance filtering, event
segmentation, attribute extraction, and deduplication. It has three parts:

1. Every numeric threshold/parameter in the code, where it lives, and why it's set where it is.
2. What every test in `tests/` actually checks.
3. Real success/failure results from deliberately adversarial cases added to
   `data/synthetic/whatsapp_export.txt` (messages 27–37), run through the real pipeline —
   not hypothetical, these are the actual outputs.

All of this was tuned/observed on a 37-message synthetic dataset. That's small. Treat every
number below as a starting point to re-check once we have more data, not a validated constant.

## 1. Thresholds and parameters

| Parameter | Value | File | What it controls | Why this value |
|---|---|---|---|---|
| Relevance decision | argmax of 2 candidate labels (no numeric cutoff) | `src/relevance_classifier.py` | Whether a message is `relevant` vs `irrelevant` | Zero-shot classification always picks a top label — there's no built-in "unsure" option. The `confidence` score is reported but nothing currently *acts* on it. |
| Relevance F1 target | ≥ 0.8 | `tests/test_relevance_classifier.py` (`test_real_dataset_meets_f1_threshold`) | Pass/fail gate for the relevance-filtering test | Carried over directly from the Week 5 plan's target metric. |
| Confidence "borderline" cutoff (discussed, not enforced) | ~0.6 | README only | Where we'd route to human review *if* that step existed | Chosen by eyeballing the one real false positive we have (0.51 confidence) — not derived from enough data to trust yet. **Not implemented in code.** |
| Segmentation time window | `max_gap_minutes=10` | `src/segment.py` (`segment_events`) | Two relevant messages from the same sender merge into one event if ≤10 min apart | Arbitrary starting point — picked to catch the "forgot to mention X, sending a follow-up a minute later" case without merging across a whole workday. Cases 29/30 and 31/32 below show this cuts both ways. |
| Dedup similarity threshold | `threshold=0.75` | `src/dedup.py` (`find_duplicates`) | Two events flagged as duplicates if cosine similarity ≥ this | Set after observing the real duplicate pair (goat distribution, messages 3+4 vs. 26) score 0.80, with the next-highest *unrelated* pair at only 0.48 — picked 0.75 for a comfortable margin below the true positive. See case 37 below for where this threshold still gets fooled. |
| Phone number regex | `\+?\d[\d\-\s]{7,}\d` | `src/anonymize.py` | What counts as a redactable phone number | Matches digits separated by spaces/dashes with an optional leading `+`. Does **not** match dot-separated numbers (`0712.345.678`) — see case 28. |
| Age regex | `\(?\s*age\s*\d{1,3}\s*\)?` or `\d{1,3}\s*years?\s*old` | `src/anonymize.py` | What counts as a redactable age | Covers the two phrasings used in our synthetic data; not a general solution. |
| NER entity type redacted | `PER` only | `src/anonymize.py` | Locations (`LOC`) and organizations (`ORG`) are deliberately **not** redacted | Program sites aren't personal addresses, and M&E reporting needs a location field. Documented trade-off, not an oversight — see README "Known limitations". |

## 2. What each test file checks

### `tests/test_relevance_classifier.py`
| Test | Checks |
|---|---|
| `test_parse_whatsapp_export` | The WhatsApp-export regex parser extracts `id`/`date`/`time`/`sender`/`text` correctly from two sample lines. No model involved. |
| `test_evaluate_perfect_predictions` | `evaluate()`'s precision/recall/F1 math is correct when predictions exactly match gold labels. No model involved. |
| `test_evaluate_catches_false_positive_and_false_negative` | `evaluate()` correctly counts a false positive and false negative and computes precision=recall=0 for that case. No model involved. |
| `test_real_dataset_meets_f1_threshold` | **Integration test.** Runs the real `facebook/bart-large-mnli` model on the full synthetic dataset and asserts F1 ≥ 0.8 against `gold_labels.jsonl`. |

### `tests/test_anonymize.py`
| Test | Checks |
|---|---|
| `test_phone_and_age_redacted_without_model` | The phone/age regexes redact the two known phrasings. Pure regex, no model. |
| `test_person_names_fully_redacted` | **Integration.** A clearly-tagged name ("Mama Achieng") is fully redacted to `[PERSON]`, no fragment of the original name remains. |
| `test_location_is_not_redacted` | **Integration.** A place name ("Kakamega") survives anonymization untouched (by design — see thresholds table). |
| `test_name_split_into_multiple_entities_does_not_corrupt_text` | **Regression test** for the overlapping-span bug found while adding case 31/37 below (see §3) — asserts the exact corrected output, not just "no crash". |

### `tests/test_segment.py`
| Test | Checks |
|---|---|
| `test_merges_same_sender_within_gap` | Two same-sender messages 2 minutes apart merge into one event with both message IDs. No model. |
| `test_does_not_merge_different_sender` | Two different-sender messages close in time stay as separate events. No model. |
| `test_does_not_merge_outside_time_window` | Two same-sender messages outside the 10-minute window stay separate. No model. |
| `test_empty_input` | Empty input returns an empty event list. No model. |

### `tests/test_extract.py`
| Test | Checks |
|---|---|
| `test_to_iso_date` | Date-format conversion (`DD/MM/YYYY` → `YYYY-MM-DD`) is correct. |
| `test_livelihood_distribution_with_quantity_and_outcome` | A clean example extracts the right activity type, quantity, outcome, and date together. |
| `test_training_with_location` | A clean example extracts a known location correctly alongside activity type and quantity. |
| `test_unmatched_text_falls_back_to_other` | Text matching no activity keyword falls back to `"other"` with `quantity`/`location` as `None`, rather than guessing. |

### `tests/test_dedup.py`
| Test | Checks |
|---|---|
| `test_near_duplicate_events_flagged` | **Integration.** Two genuinely-duplicate texts score above threshold and get flagged; an unrelated third event doesn't. |
| `test_single_event_not_flagged` | A single event with nothing to compare against is never flagged as a duplicate (no index-out-of-range, no model needed). |

### `tests/test_pipeline.py`
| Test | Checks |
|---|---|
| `test_full_pipeline_on_synthetic_data` | **Full integration test.** Runs anonymize → filter → segment → extract → dedup end-to-end on the real dataset and asserts: (1) messages 3+4 merge into one event, (2) message 26 is flagged as a duplicate of that event, (3) that event's extracted activity type and quantity are correct, (4) every parsed message ends up in exactly one of `events`/`irrelevant_messages` (no message silently dropped or duplicated). |

Run everything with:
```bash
python -m pytest tests/ -v
```

## 3. Adversarial cases (messages 27–37): what actually happened

These were added on purpose to find where the baseline breaks, not to make everything pass.
Results below are from the actual pipeline run (`outputs/pipeline_output.json`), not predictions.

| # | Message | Targets | Expected failure mode | Actual result | Verdict |
|---|---|---|---|---|---|
| 27 | "Sorted the usual with the Odhiambo family, all good." | Relevance filter | Too vague to trigger the model's "field activity" hypothesis | Classified `irrelevant` at 0.83 confidence | **Failure (as expected)** — a real false negative, genuinely relevant but too terse to catch. |
| 28 | "...contact number...to 0712.345.678..." | Anonymization (phone regex) | Dot-separated phone number doesn't match `PHONE_RE` | Number left in plain text; **and** the extraction regex then grabbed `"678 after her follow-up visit"` as a bogus `quantity` | **Failure** — worse than expected: an anonymization gap directly caused a nonsense downstream extraction. Good example of error cascading. |
| 29+30 | "Delivered mosquito nets to 6 households..." then (3 min later, same sender) "Also started a new training session for 12 farmers..." | Segmentation | Same-sender, within-window rule wrongly merges two *unrelated* activities into one event | Merged into one event; `activity_type` extraction only picked up `"training"` (the second sentence), losing the mosquito-net distribution entirely | **Failure (as expected)** — confirms segmentation has no topic awareness, exactly the documented limitation. |
| 31+32 | "Checking on the Otieno household's new goat shed..." then (12 min later, same sender) "Materials for the shed were enough, roofing to finish tomorrow." | Segmentation | Real continuation of one event, but 12 min > the 10-min window, so it wrongly splits into two | Stayed as two separate events, second one extracted as `activity_type: "other"` | **Failure (as expected)** — the opposite failure mode from 29/30: same rule, now too strict. |
| 33 | "Helped repair the community borehole pump so it works again." | Attribute extraction | Real activity with no matching keyword rule | `activity_type: "other"`, no hallucinated guess | **Partial** — correctly avoids hallucinating a wrong category (good), but still a coverage gap (bad): a real activity reported as `"other"`. |
| 34 | "Gave out twenty two blankets to families..." | Attribute extraction (quantity) | Quantity written in words, not digits, won't match `QUANTITY_RE` | `activity_type: "distribution"` correct (via "gave out"), but `quantity: None` | **Partial failure** — activity type right, quantity silently missed. |
| 35 | "The family refused the proposed support and asked us not to return." | Attribute extraction (outcome) | No keyword in `OUTCOME_KEYWORDS` covers a refusal/negative outcome | `outcome: "unknown"` | **Failure (as expected)** — a concerning outcome that should probably get flagged ends up indistinguishable from "we just don't know." |
| 36 | "Circling back — the goat handover at the Achieng place went well earlier today." | Deduplication | Heavily paraphrased true duplicate of the messages-3/4 event, little shared vocabulary | Similarity scored only **0.296** (vs. 0.75 threshold) — not flagged | **Failure (as expected)** — a real duplicate missed because embedding similarity needs more lexical overlap than this paraphrase has. |
| 37 | "Visited the Oduya family, gave 2 goats as part of the livelihood program, they were thrilled." | Deduplication | A genuinely *different* beneficiary, but structurally near-identical phrasing to the Achieng event — and anonymization strips the one distinguishing detail (the name) | Flagged as `duplicate_of: evt_0001` at **0.81** similarity — wrongly merged with a different real event | **Failure** — and an important one: anonymization and deduplication actively work against each other here. Removing the name (correct, for privacy) removed the only thing that told two different real events apart. |

**Bugs found and fixed while building these cases** (not listed above as "failures" because they're
now fixed, with regression tests): `dslim/bert-base-NER` sometimes returns one name as 2–3 separate
adjacent `PER` entities (e.g. "Otieno" → "O" + "##tien" + "##o"), which corrupted the output text
(`"[PERSON]N]N]"`) once each fragment was word-boundary-extended and redacted independently. Fixed
by merging overlapping spans before redacting (`src/anonymize.py`, `_merge_overlapping_spans`), with
a regression test in `tests/test_anonymize.py`.

**Interface mismatch found integrating with Ronav's `activity_baseline.py`:** his script expects
the input JSON to *be* a bare list of event records (`for message in messages: ...`), but
`pipeline.py`'s main output is `{"events": [...], "irrelevant_messages": [...]}` — running his
script directly on `outputs/pipeline_output.json` crashes (`TypeError: string indices must be
integers`, from iterating the dict's keys as if they were records). Fixed by having `pipeline.py`
also write a bare events list via `--events-output` (`outputs/events_for_review.json`), rather than
changing his script to unwrap a format it was never shown. Regression test:
`tests/test_integration.py`.

**Finding, not yet a decision, from running the real full chain:** feeding our 22 real events
through `activity_baseline.py` flags **20 of 22 (91%)** as `needs_review`, almost entirely because
`location` is `None` (our extractor only recognizes 2 hardcoded place names — see §1). If "flag
anything missing a field" means 91% of records always need human review, the review queue isn't
actually triaging anything. This also can't tell "field is null" apart from "field has a wrong
value" — e.g. event with the bogus `"678 after her follow-up visit"` quantity (case 28) only got
flagged for low relevance confidence, not for the garbage quantity. Also: his grouping-by-
`activity_key` mechanism has nothing to group on yet, since `pipeline.py` doesn't produce an
`activity_key` field — every event passes through as its own singleton group (confirmed: 22 in,
22 out). None of this is "wrong" so much as **unfinished coordination** between what Roland's
extraction produces and what Ronav's review layer expects — worth a real conversation, not a
unilateral fix to either side. Flagged for Week 7.

**Bug found integrating with Sathvik's ingestion (`src/parser.py`):** its day-first/month-first
auto-detection heuristic needs at least one date in the file where day or month is > 12 to tell
which convention is in use. Every date in our synthetic export has both ≤ 12 (`03/02/2026`), so
there's nothing ambiguous for it to detect, and it silently defaulted to month-first — reading
"3 February" as "2 March" for every single message. Fixed by adding an optional `dayfirst`
parameter to `parse_chat()` that overrides the heuristic when the caller already knows the
convention (ours does: `src/pipeline.py` passes `dayfirst=True`), without changing the default
behavior his own tests rely on. Regression test: `tests/test_parser.py::test_dayfirst_override`.
This is also the main reason `src/pipeline.py` now calls `parser.parse_chat()` directly instead of
keeping its own separate parser — running the actual shared ingestion code is what surfaced this.

**Takeaway for Week 7/8:** the two deduplication failures (36, 37) point at the same root cause from
opposite directions — pure text-similarity dedup can't reliably tell "same event, different words"
apart from "different event, same words" once anonymization has removed the one detail that would
disambiguate them. That's worth a real design conversation before Week 8's hidden-test stress test,
not just a threshold tweak.
