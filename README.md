# CSEN 166 — Term Team Project for Social Impact (TTP4SI)

**Team:** D3  
**Challenge:** Dandelion Africa  
**Team members:** Roland Breise, Sathvik Kumar, Ronav Poonekar  
**Repository:** [Dandelion3](https://github.com/SCU-CSEN166/Dandelion3)  
**Last updated:** 2026-10-30

> **How to use this README:** Maintain this single file throughout the project. Complete the Week 5 project plan first, then update the appropriate weekly section before each milestone. Be specific and honest: describe what you attempted, what worked, what did not work, and what evidence supports your conclusions. For every week, document each member's actual contribution. If a task was not completed, explain the blocker and next step. Do not include private or sensitive data in this public README.

## Remember - Best Practices

### Data access and privacy
We have **no access to Dandelion Africa's real WhatsApp data** because Kenya's data-sharing policy does not allow exporting it to the US.

All project data is **synthetic**. The team writes example activity reports and casual messages. Team members, friends, and family record the voice notes, with TTS as a fallback. We aim to include different English accents and especially women's voices to reflect the user base.

All names, ages, locations, and phone numbers in these messages are made up. We still anonymize them before including them in README examples. No real PII, credentials, or API keys are committed. Synthetic audio and text are stored in `data/synthetic/`; raw recordings are gitignored, and only anonymized examples are checked in.

### Known limitations
- Sathvik's message parser and Roland's relevance classifier are implemented; ASR, anonymization, event segmentation, attribute extraction, dedup, and confidence routing are not built yet.
- The relevance classifier has only been tested on 24 team-written synthetic messages — too small to trust the 0.957 F1 as a real estimate yet.
- Our synthetic data may not reflect messages that mix English with Swahili or Dholuo, background noise, overlapping speakers, or the activity categories Dandelion Africa uses.
- We can only test accents spoken by the volunteers we find (no audio data yet — text only so far).

## Week 5 — Scope: Project Plan

**Deliverable during Week 6 Lab:** A project plan (write in this README file) describing your problem formulation, proposed approach, each team member's responsibilities, proposed baseline models with Hugging Face links, proposed evaluation metrics, proposed data with links, and goals for each week.

### 1. Problem formulation and social impact
- **Problem:** Dandelion Africa field/community health workers report on their activities (home visits, trainings, livelihood distributions, patient follow-ups) informally in WhatsApp group chats, mixing text and voice notes with unrelated chit-chat. Program/M&E staff currently have to read through this manually to know what happened. We will build a pipeline that turns WhatsApp text and voice notes into structured records of field activities and outcomes.
- **Intended users:** Dandelion Africa program/M&E (monitoring & evaluation) staff who need a structured activity log instead of scrolling through raw chat history.
- **Inputs and outputs:** Input = a WhatsApp chat export (ordered messages with sender, timestamp, text and/or voice-note audio). Output = a list of structured event records. Example: input message "Visited Mama A.'s homestead today, gave 2 goats as part of the livelihood program, she was very happy" (voice note, transcribed) → output `{event_type: "livelihood_distribution", date: "2026-02-03", beneficiary: "Mama A. (anonymized)", quantity: "2 goats", outcome: "positive", confidence: 0.82, source_msg_id: 145}`.
- **Success criteria:** The pipeline correctly identifies which messages describe real activities (high precision/recall), extracts the right structured fields for those activities, never leaks PII into the output, and flags low-confidence or ambiguous extractions for human review.
- **Constraints and risks:** We do not have access to real WhatsApp data. If speech recognition gets words wrong, the extracted information may also be wrong. The LLM might add details that weren't in the original message. Anonymization needs to catch names and phone numbers so they do not remain visible in the output. Our synthetic messages may differ from the messages workers actually send.

### 2. Proposed technical approach
- **Pipeline:**
  1. **Parse** — parse the WhatsApp export (chat log + audio attachments).
  2. **ASR** — transcribe voice notes to text. (Whisper)
  3. **Anonymization** — detect and redact PII (names, ages, locations, phone numbers) in both text and transcripts. (Bert)
  4. **Relevance filtering** — classify each message as a real field-activity report vs. irrelevant chit-chat. (Bart)
  5. **Event segmentation** — group related messages (same thread/time window) into one activity "event." (Bart)
  6. **Attribute extraction** — extract the activity type, date, location, quantity, and outcome from each event using a fixed output format. (llama)
  7. **Deduplication** — detect multiple messages reporting the same underlying event.   (llama)
  8. **Confidence estimation** — combine ASR confidence, extraction self-consistency, and rule checks into a per-record confidence score. (bart)
  9. **Human verification queue** — route low-confidence/ambiguous records for review (exported as a simple spreadsheet).
- **Baseline(s):** Simplest end-to-end version: keyword/regex relevance filter + Whisper (small) for ASR + a single few-shot LLM prompt that does filtering, anonymization, and structured JSON extraction in one pass, with no dedup or confidence scoring yet.
- **Proposed models/tools:**
  - [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) — Transcribes voice notes. We chose large-v3 for better accuracy on accented English. We will use `whisper-small` if large-v3 runs too slowly or needs more computing resources than we have.
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) — speaker diarization if a voice note or export segment has multiple speakers.
  - [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) — NER-based detection of names/locations for anonymization (combined with regex for phone numbers/ages). We also evaluated Microsoft Presidio as a replacement for this approach or an additional tool.
  - [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) — zero-shot classifier for the relevance-filtering step before we have enough labeled data to fine-tune.
  - [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — sentence embeddings for similarity-based deduplication of repeated reports.
  - [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) — instruction-following LLM for schema-constrained structured extraction.
- **Alternative approach:** If Whisper struggles with accented speech, we will try fine-tuning it on accented-English recordings such as [AfriSpeech-200](https://huggingface.co/datasets/intronhealth/afrispeech-200). If the LLM keeps adding details that are not in the messages, we will try stricter output constraints using Outlines or a JSON-schema generation library. Another option is to fine-tune a smaller model for extraction.

### 3. Proposed data
| Dataset/source | Link or access instructions | Purpose | Size / languages | Privacy or usage considerations |
|---|---|---|---|---|
| Team-authored synthetic WhatsApp text | Created by team (scripts in `data/synthetic/`) | Training/development/evaluation | ~50–100 synthetic messages/threads to start, English, mix of relevant activity reports + chit-chat | All names/phone numbers/locations are fabricated; still passed through the anonymization step before being used as examples |
| Team-recorded synthetic voice notes | Recorded by team members/friends/family (or TTS fallback) | Development/evaluation of ASR + full pipeline | Target: several accents/voices, majority women speakers, English | Volunteers agree to record fictional messages. Recordings contain no real personal disclosures. |
| [AfriSpeech-200](https://huggingface.co/datasets/intronhealth/afrispeech-200) | Hugging Face dataset | ASR robustness reference/validation (African-accented English) | ~200 hrs, African-accented English | Public research dataset; used only as an ASR benchmark, not Dandelion-specific content |
| [Mozilla Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Hugging Face dataset | Supplementary ASR validation, accent diversity | Multilingual, accent-tagged subset usable | Public, consented crowd-sourced dataset |

- **Data preparation:**
  - Write a script to create synthetic WhatsApp threads with senders, timestamps, and text or audio.
  - Record the audio or generate it with TTS.
  - Manually label the development set for relevance, event type, attributes, and fake personal information.
  - Keep raw recordings out of version control and commit only anonymized examples.
- **Data limitations:** Our synthetic data may not reflect real messages that mix Swahili or Dholuo with English, background noise, or Dandelion Africa's activity categories. We are using the challenge description until we can check the terms the organization actually uses.

### 4. Proposed evaluation
| Metric / test | What it measures | How it will be calculated | Target / success criterion |
|---|---|---|---|
| Relevance F1 | Quality of filtering real activity reports from chit-chat | Precision/recall/F1 vs. gold-labeled dev messages | F1 ≥ 0.8 on dev set |
| Word Error Rate (WER) | ASR transcription quality | WER of Whisper transcript vs. gold transcript on recorded dev audio | Initial target: WER ≤ 20%. We will also check how transcription errors affect extraction accuracy. |
| Field-level extraction accuracy | Correctness of structured attributes extracted per event | Per-slot exact/partial match vs. gold structured record | ≥ 0.75 per-slot accuracy on dev set |
| PII recall (anonymization) | Whether all injected PII is caught | % of injected fake names/ages/locations/phone numbers successfully redacted | As close to 100% as possible |
| Deduplication precision/recall | Whether repeated reports of the same event are correctly merged | Pairwise match vs. gold duplicate-pairs labels | To be set once we have enough duplicate examples in the dev set |
| Confidence calibration | Whether low-confidence flags correlate with actual errors | AUROC of confidence score vs. correctness on dev set | Correct extractions should generally have higher confidence scores than incorrect ones. |

- **Evaluation examples:** We will build our own labeled dev set (~30–50 synthetic threads, mixing relevant/irrelevant messages, injected PII, and some duplicate reports) and have each team member independently label a subset, then compare our labels and agree on the final answers.
- **Human verification:** Team members check each other's labels before finalizing them. Outputs with low confidence are saved for human review.
- **Failure cases to watch:** (1) ASR mishearing accented speech and causing the wrong activity type/quantity to be extracted; (2) the anonymization model missing a name because it is misspelled or transcribed incorrectly; (3) the LLM adding a detail that was not in the message (e.g., a quantity or date not actually present in the message).

### 5. Team responsibilities
| Member | Primary responsibility | Specific Weekly contribution | Planned next task |
|---|---|---|---|
| Roland Breise | Classifying & creating synthetic data | Drafted week 5 project plan and set up the shared repo | start week 6 baseline work |
| Sathvik Kumar | Ingesting data | Helped draft week 5 project plan | start week 6 baseline work |
| Ronav Poonekar | data formatting & testing | Helped draft week 5 project plan | start week 6 baseline work |

---

## Week 6 — Build: Working Baseline

**Deliverable during Week 7 Lab:** A runnable baseline, setup/run instructions, and 1–2 example inputs and outputs. Update this section with what worked, what failed, and individual contributions.

### Implementation status

> This section covers **Roland's scope** (synthetic data + relevance classifying). Sathvik's ingestion step and Ronav's data formatting/testing step are not implemented yet — see their rows below.

- **What runs end-to-end now?** `src/relevance_classifier.py` reads a WhatsApp-export-style text file, parses it into individual messages, and classifies each one as a real field-activity report ("relevant") or chit-chat ("irrelevant") using zero-shot classification — this is pipeline step 4 from the Week 5 plan. No ASR, anonymization, segmentation, extraction, or dedup yet (outside Roland's scope this week).
- **How to run it:**
  ```bash
  cd Dandelion3
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python src/relevance_classifier.py \
    --input data/synthetic/whatsapp_export.txt \
    --output outputs/relevance_predictions.jsonl \
    --gold data/synthetic/gold_labels.jsonl
  ```
- **Model(s) and data actually used:** [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) (zero-shot classification, as proposed in the Week 5 plan), run on `data/synthetic/whatsapp_export.txt` — 24 team-authored synthetic messages (12 relevant / 12 irrelevant) with `data/synthetic/gold_labels.jsonl` as the gold labels.
- **Example 1:** Input: `"Visited Mama Achieng's homestead today, gave 2 goats as part of the livelihood program. She was 67 years old and very happy with the support."` → Output: `relevant` (confidence 0.942)
- **Example 2:** Input: `"anyone have a good recipe for ugali, want to try something new tonight"` → Output: `irrelevant` (confidence 0.886)

### Attempts and outcomes
| Attempt / change | Worked? (Yes / Partly / No) | Evidence (output, metric, link) | What we learned / next step |
|---|---|---|---|
| Zero-shot relevance classifier (`facebook/bart-large-mnli`) on 24 synthetic messages | Yes | Precision 0.917, recall 1.0, F1 0.957 against `gold_labels.jsonl` (`outputs/relevance_predictions.jsonl`) | Already above the Week 5 target (F1 ≥ 0.8) on this small set. The one false positive was a "🙏 blessings, stay safe" message classified `relevant` at only 0.51 confidence — right at the decision boundary, so a confidence threshold (e.g. route anything < 0.6 to human review) should catch this kind of borderline case instead of silently misclassifying it. Next: grow the synthetic set past 24 messages and re-check F1 holds up. |

### Individual contributions during that week
| Member | What I personally implemented/tested | Result (worked / partly / failed) | Evidence (commit, PR, experiment) | Blocker / next step |
|---|---|---|---|---|
| Roland Breise | Wrote the synthetic WhatsApp dataset (`data/synthetic/whatsapp_export.txt` + gold labels) and the relevance classifier (`src/relevance_classifier.py`, zero-shot BART-MNLI) | Worked | F1 0.957 on the 24-message dev set; see `outputs/relevance_predictions.jsonl` | No blocker. Next: expand the synthetic dataset (more messages, more activity types, more borderline cases) and add a confidence-threshold rule for the human-review queue |
| Sathvik Kumar | Added a WhatsApp message parser (`src/parse_messages.py`) and sample data (`data/synthetic/sample_messages.json`) | Worked | [data/synthetic](https://github.com/SCU-CSEN166/Dandelion3/tree/main/data/synthetic) | Implement AI models and tests |
| Ronav Poonekar | [Fill in — data formatting & testing] | [Status] | [Link] | [Next step] |

---

## Week 7 — Evaluate: Results Checkpoint

**Deliverable during Week 8 Lab:** Preliminary evaluation results, metrics, development-set details, and 2–3 representative successes or failures. Include what you will improve next.

### Evaluation setup
- **Development data:** [Source, size, selection, annotations, link or access instructions]
- **Metrics and rationale:** [Why these metrics reflect the intended use]
- **Comparison:** [Baseline versus current approach, if different]

| System / version | Metric 1 | Metric 2 | Notes |
|---|---:|---:|---|
| Baseline | [Value] | [Value] | [Notes] |
| Current | [Value] | [Value] | [Notes] |

### Successes, failures, and interpretation
| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
|---|---|---|---|---|
| [Case] | [Expected] | [Observed] | [Status] | [Hypothesis] |

- **What improved?** [Evidence]
- **What did not improve?** [Evidence]
- **Next experiment:** [Specific change and how you will measure it]

### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Next step] |

---

## Week 8 — Stress-Test: Reliability on Hidden Data

**Deliverable during Week 9 Lab:** Hidden-test results (as provided by the TA), analysis of at least three important failure cases, and a prioritized improvement plan.

- **Submitted system version / commit:** [Commit hash]
- **Hidden-test metrics and leaderboard result:** [Results, when available]
- **Difference from development results:** [Compare; do not assume the hidden set is identical]

| Failure case / category | What went wrong | Possible cause | Severity / user impact | Proposed fix |
|---|---|---|---|---|
| [Case 1] | [Observation] | [Hypothesis] | [Impact] | [Fix] |
| [Case 2] | [Observation] | [Hypothesis] | [Impact] | [Fix] |
| [Case 3] | [Observation] | [Hypothesis] | [Impact] | [Fix] |

### Changes attempted after stress-testing
| Change | Worked? (Yes / Partly / No) | Evidence | Keep, revise, or revert? |
|---|---|---|---|
| [Change] | [Status] | [Result] | [Decision] |

### Individual contributions during that week
| Member | Work completed | Result (worked / partly / failed) | Evidence | Next step |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Next step] |

---

## Week 9 — Responsible AI, Peer Testing, and Final Delivery

**Deliverable during Week 10 Lab:** Document peer-testing feedback, Responsible AI concerns, changes made in response, and the final runnable system. Submit the final project report as instructed.

### Peer testing
- **Team evaluated:** [Team ID]
- **One thing that worked well:** [Evidence]
- **One meaningful failure:** [Reproduction steps or example]
- **One actionable recommendation:** [Specific fix]
- **Feedback received from another team:** [Summary]

### Responsible AI assessment
- **Potential harms or affected users:** [Who could be affected and how?]
- **Privacy and data handling:** [How sensitive information is protected]
- **Reliability and unsupported claims:** [How outputs are verified or uncertainty is communicated]
- **Fairness, accessibility, and usability:** [Relevant concerns and tests]
- **Human oversight:** [Where a human should review or correct outputs]
- **Remaining limitations:** [What should not be trusted or deployed yet]

### Final improvements and outcomes
| Feedback / issue | Change made | Worked? (Yes / Partly / No) | Evidence | Remaining limitation |
|---|---|---|---|---|
| [Issue] | [Change] | [Status] | [Link or metric] | [Limitation] |

### Individual contributions during that week
| Member | Final implementation / evaluation / documentation contribution | Result (worked / partly / failed) | Evidence | Reflection / remaining work |
|---|---|---|---|---|
| [Name] | [Specific work] | [Status] | [Link] | [Reflection] |



### Final submission checklist
- [ ] Code and final system/prototype are in the repository
- [ ] Setup and run instructions are complete and tested
- [ ] Data sources, model versions, and dependencies are documented
- [ ] Evaluation results are reported and compared with the baseline
- [ ] Important failures and limitations are explained
- [ ] Hidden-test and peer-feedback findings are addressed
- [ ] Responsible AI considerations and human oversight are discussed
- [ ] Every team member's contributions are documented for every week

**Final commit / release:** [Commit hash or tag]

---

## Setup and Reproducibility (keep updated)

### Requirements
Python 3.9+. Libraries: `transformers`, `torch` (see `requirements.txt`). No GPU required — `facebook/bart-large-mnli` runs on CPU (or Apple Silicon MPS) for a dataset this small; the model (~1.6GB) downloads automatically from Hugging Face on first run.

### Installation
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run the system
```bash
python src/relevance_classifier.py \
  --input data/synthetic/whatsapp_export.txt \
  --output outputs/relevance_predictions.jsonl
```

### Run evaluation
```bash
python src/relevance_classifier.py \
  --input data/synthetic/whatsapp_export.txt \
  --output outputs/relevance_predictions.jsonl \
  --gold data/synthetic/gold_labels.jsonl
```

### Run tests
```bash
python -m pytest tests/ -v
```
The first three tests check the WhatsApp parser and the precision/recall/F1 scoring logic directly (fast, no model needed). The last test is a real integration check — it runs `facebook/bart-large-mnli` on `data/synthetic/whatsapp_export.txt` and fails if F1 drops below the Week 5 target (0.8).

### Repository structure
```text
Dandelion3/
├── README.md
├── requirements.txt
├── data/
│   └── synthetic/        # Roland: synthetic WhatsApp text + gold labels
├── src/
│   └── relevance_classifier.py   # Roland: relevance-filtering baseline
├── tests/
│   └── test_relevance_classifier.py   # Roland: parser + scoring + F1-threshold tests
└── outputs/
    └── relevance_predictions.jsonl
```


