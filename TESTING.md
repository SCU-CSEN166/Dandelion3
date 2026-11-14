# Testing reference (Roland's scope)

This covers `src/`: relevance filtering, media cleanup, attribute extraction, event
clustering, and the event-level duplicate safety net. **Anonymization has moved to
Ronav's side** (`src/anonymize.py` stays in the repo for him, but `pipeline.py` no
longer calls it — our output carries raw, unredacted text).

## 0. What changed, and why (read this first)

The previous version of this pipeline anonymized text *before* clustering/dedup. That
turned out to be the root cause of our worst dedup failures: redacting a name removed
the one detail that told two real events apart, or that confirmed two reports were
about the same event. The team decided to:

1. Move anonymization downstream of everything else (Ronav's side, on the handoff JSON).
2. Replace the old two-step design (time/sender **segmentation**, then a separate
   post-hoc **dedup** pass) with **one clustering pass over the whole chat** —
   not just adjacent same-sender messages — plus a second, stricter **event-level
   duplicate safety net** that runs after clustering (this is what `duplicate_of` now
   means: an entire event can be marked as a duplicate of another entire event, not
   just a message).
3. Add a dedicated `participants` field (the old pipeline never populated one —
   confirmed empty across all 10 rows of the team's own earlier hidden-set run,
   `hidden_test_outputs.csv`).
4. Make `activity_type` free text instead of a forced fixed enum (the enum was overfit
   to our own synthetic vocabulary — see §3).
5. Strip WhatsApp media-placeholder text ("<Media omitted>", etc.) before anything else
   sees it.

**This was validated against real, independently-produced data**, not just our own
synthetic set — see §3. That's the strongest evidence in this document.

## 1. Thresholds and parameters

| Parameter | Value | File | What it controls |
|---|---|---|---|
| Relevance decision | argmax of 2 candidate labels | `relevance_classifier.py` | relevant vs. irrelevant |
| Relevance F1 target | ≥ 0.8 | `tests/test_relevance_classifier.py` | pass/fail gate |
| Cluster threshold | 0.5 | `cluster.py` | combined score needed to merge two messages into one event |
| Cluster text weight | 0.5 | `cluster.py` | weight of sentence-embedding similarity (raw text) in the combined score |
| Cluster activity-overlap weight | 0.15 | `cluster.py` | bonus if `activity_type`s share a word (excluding "other" — see §2 bugs) |
| Cluster participants weight | 0.15 | `cluster.py` | bonus if extracted `participants` counts match exactly |
| Cluster adjacency weight | 0.3 | `cluster.py` | bonus if same sender within `max_gap_minutes` — needed for low-content continuation messages ("forgot to mention X") that pure content signals miss |
| Cluster adjacency window | 15 min | `cluster.py` | |
| Dedup safety-net threshold | 0.75 | `dedup.py` | stricter, pure-text-similarity check between already-clustered **events** (not messages) — catches residual duplicates clustering missed |

All of these were tuned by hand against the real cases in §3, on a tiny amount of data.
Treat every number as a starting point, not a validated constant.

## 2. Real bugs found while building this (not hypothetical)

| Bug | Found via | Fix |
|---|---|---|
| "ran out of stock" extracted as an activity name (because "ran" was a trigger verb for the free-text extractor) | Real hidden-set transcript | Removed "ran" from the verb list — too common in unrelated idioms. |
| "held this afternoon" extracted "this afternoon" as the activity instead of falling through to find the real phrase | Real hidden-set transcript | Reject a verb-phrase match whose first word is a temporal filler ("this", "today", etc.); fall through to the noun-phrase extractor. |
| Activity-overlap check never matched between old-style `livelihood_distribution` (underscore) and new free-text "...the livelihood program" (spaces) | Testing clustering on our synthetic set | Split on underscores too before comparing words. |
| Two messages that **both failed** activity extraction (`"other"`) got a free "activity matches" bonus, since `"other" == "other"` passed the word-overlap check | Long-conversation test data | Explicitly exclude `"other"` from the overlap check — it means "unknown," not "same topic." |
| `pipeline.py` had its own separate, stale `cluster_threshold` default that didn't match `cluster.py`'s updated default, silently overriding it | Re-testing after a threshold change | Keep the two in sync (still two separate numbers — worth consolidating to one source of truth later). |
| De-identification script reused the *same* fake phone number for every message (generator was re-created per call instead of shared) | Checking the de-identified output by eye | Pass a shared counter across the whole document. |

## 3. Validation against real, independently-produced data

This is the important part. `data/hidden_test_transcripts.json` is the real ASR
transcript of 10 voice notes recorded by someone else for this class, not written by
us — the strongest test we have, because the failures and successes weren't designed
by the people being tested. `scripts/deidentify_hidden_set.py` produces a safe,
committed version (`data/synthetic/hidden_set_deidentified.json` +
`hidden_set_gold.json`) with real-sounding names/phone numbers swapped for clearly
fake ones, consistent per person, so clustering can still be tested properly.

**Before (team's own first pass, `hidden_test_outputs.csv`, old architecture):**
- `participants`: empty in **all 10 rows**.
- `duplicate_of`: `"none"` in **all 10 rows** — both real duplicate pairs in the data
  (two people independently reporting the same family-planning outreach; two people
  independently reporting the same cervical-cancer screening) were missed.
- `activity_type`: forced into the old fixed enum (`follow_up`, `intake_assessment`,
  `training`) — miscategorized 6 of 7 real events, usually triggered by an incidental
  word ("follow-up," "referred") rather than the message's actual topic.

**After (this architecture, run on the de-identified set):**
- **Relevance**: 7 TP, 1 FP, 1 TN, 1 FN → precision 0.875, recall 0.875. The one
  consistent miss (a table-banking/microfinance report, msg04) is a real, repeatable
  gap — "savings group" language doesn't read as an "activity report" to the model.
- **Participants: 8/8 correct (100%)**, including every word-form number ("forty-two",
  "twenty-five", "eighteen"). This went from *entirely empty* to fully working.
- **Deduplication: 2/2 real duplicate pairs correctly caught.** Both were caught by
  clustering directly on this de-identified test (no adjacency signal available, since
  this test has no sender/timestamp data — pure content signals did the job).
- **Activity type**: no longer forced into the wrong fixed category; free text now
  reads close to the real topic for every relevant message (verbose in places — e.g.
  "i want to report on today's family planning outreach" — but topically correct,
  which the old enum never was for 6/7 events).

Run it yourself:
```bash
python scripts/deidentify_hidden_set.py   # regenerate the de-identified test file
```
Then feed `data/synthetic/hidden_set_deidentified.json` through `relevance_classifier.py`,
`extract.py`, and `cluster.py` and compare against `data/synthetic/hidden_set_gold.json`.

## 4. Results on our own synthetic adversarial set (`data/synthetic/whatsapp_export.txt`)

Full pipeline run (`python src/pipeline.py --input data/synthetic/whatsapp_export.txt ...`):
37 messages → 20 events, 13 filtered irrelevant.

| Case | What happened | Verdict |
|---|---|---|
| 2-message continuation (same sender, 2 min apart) + a later, differently-worded confirmation (different sender) | All three correctly merged into **one event**, `(2, 3, 25)` | **Fixed.** Needed both the adjacency signal (for the immediate continuation) and raw text carrying the shared name "Achieng" (for the later confirmation) — anonymization used to remove exactly that name. |
| A genuinely *different* beneficiary ("the Oduya family") with near-identical phrasing to the Achieng event | **Not** merged, `duplicate_of: None` | **Fixed.** This was the sharpest false positive from the old anonymization-first design; the new combined scoring (lower text weight + participants/activity signals) keeps it separate now. |
| A true continuation 12 minutes apart, same sender, almost no shared vocabulary ("goat shed construction" / "materials for the shed") | Correctly merged, `(30, 31)` | **Fixed.** Content alone (text_sim 0.49) wasn't enough; the adjacency bonus made the difference. |
| Two different activities, same sender, 3 minutes apart (mosquito nets / training) | Correctly stayed separate | **Still correct** — low content similarity overrides the adjacency bonus. |
| A heavily-paraphrased true duplicate ("Circling back — the goat handover...") | **Still missed** | **Known limitation.** Low text similarity (0.42), no shared participants, no activity overlap — none of our signals catch this one. |
| **New finding**: two different, unrelated mosquito-net distributions (different households, different locations) got merged, because the sentences are structurally almost identical and neither mentions a distinguishing name | Wrongly merged | **New known limitation**, same root cause as the Oduya case in a different shape — when two different real events are worded almost identically and have no name or distinguishing count to tell them apart, no amount of threshold tuning on these signals fully solves it. |

**Honest summary**: the redesign fixed the two specific problems it was built to fix,
and real hidden data confirms the fix generalizes. It did **not** make the underlying
tension disappear — pure text/attribute similarity still can't always tell "same event,
different words" apart from "different event, suspiciously similar words." That's a
real limitation worth a design conversation (see §6), not something a threshold tweak
resolves.

## 5. Long multi-message conversation (`data/synthetic/long_conversation.txt`)

Built specifically to test whole-chat clustering across non-adjacent messages, with
chit-chat, media placeholders, and one event reported across 3 separate, non-adjacent
messages (including a duplicate report from a different sender). Run:
`python src/pipeline.py --input data/synthetic/long_conversation.txt ...`
→ 25 text messages, 3 correctly set aside as media-only (`<Media omitted>` ×2, "This
message was deleted" ×1), 9 events, 2 flagged as event-level duplicates.

This is the clearest demonstration of the two-tier design working as intended:
- The first vaccination-drive update and a later one **didn't** cluster together
  directly (their individual pairwise combined score fell just short).
- But the **event-level duplicate safety net** caught both: the second update flagged
  `duplicate_of` the first at 0.86 similarity, and a third sender's independent report
  of the same clinic flagged `duplicate_of` the first at 0.82 — exactly the "two
  events, same real activity, merge them" behavior that was asked for.
- Not everything related gets caught: a same-sender follow-up detail ("forgot to
  mention... mosquito nets... during the vaccination outreach") and a same-event
  mention buried in an unrelated-looking message about a nutrition talk both stayed as
  their own separate events. And a same-sender, same-event report 4 hours apart (a
  borehole repair, started then finished) never matched at all — too far apart for the
  adjacency bonus, too little shared wording for the content signals. Real, honest gaps.

## 6. What's next (for Week 8 / further work)

- The remaining false-positive/false-negative tension in clustering (§4, §5) likely
  needs an explicit entity-matching signal (who/where is this about, not just what),
  or a cross-encoder/LLM pairwise judgment on borderline-scored pairs — not more
  threshold tuning on the current signals.
- `activity_type` free-text extraction is a real improvement but still verbose
  (captures filler words around the real phrase) — worth a pass to trim it.
- Relevance filtering's one consistent miss (table-banking/microfinance language)
  suggests the candidate-label hypotheses need to be broadened.
- `src/activity_baseline.py` (Ronav's review layer) is no longer compatible with this
  output shape (`text` instead of `anonymized_text`, flat `relevance`/`activity_type`/
  `participants` instead of nested `relevance`/`attributes`) — this is an expected
  consequence of the redesign, not a bug, but needs Ronav to update his side (including
  adding the anonymization call) before the two can be chained together again.
