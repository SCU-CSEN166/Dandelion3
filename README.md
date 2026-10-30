# CSEN 166 — Term Team Project for Social Impact (TTP4SI)

**Team:** D3  
**Challenge:** Dandelion Africa  
**Team members:** Roland Breise, Sathvik Kumar, Ronav Poonekar  
**Repository:** [Dandelion3](https://github.com/SCU-CSEN166/Dandelion3)  
**Last updated:** 2026-10-30

> **How to use this README:** Maintain this single file throughout the project. Complete the Week 5 project plan first, then update the appropriate weekly section before each milestone. Be specific and honest: describe what you attempted, what worked, what did not work, and what evidence supports your conclusions. For every week, document each member's actual contribution. If a task was not completed, explain the blocker and next step. Do not include private or sensitive data in this public README.

## Remember - Best Practices

### Data access and privacy
We have **no access to Dandelion Africa's real WhatsApp data**, Kenya's data-sharing policy does not allow exporting it to the US. All data used in this project is **synthetic**, created by the team: text messages written to mimic plausible field-activity reports and chit-chat, plus voice notes recorded by team members/friends/family (or TTS-generated as a fallback) to cover a range of English accents, especially women's voices, similar to the real user base. Any names, ages, locations, or phone numbers appearing in synthetic messages are themselves fabricated, never real people's data, and are additionally run through the anonymization step before being used in examples here. No real PII, credentials, or API keys are committed to this repo; synthetic raw audio/text lives under `data/synthetic/` (gitignored for raw recordings, with only anonymized derived examples checked in).

### Known limitations
- Pre-implementation: no pipeline exists yet; see the plan below.
- Synthetic data cannot fully replicate real field conditions (code-switching with Swahili/Dholuo, background noise, overlapping speakers, actual Dandelion Africa activity taxonomy).
- Accent coverage is limited to whichever voices the team can realistically record/recruit.

## Week 5 — Scope: Project Plan

**Deliverable during Week 6 Lab:** A project plan (write in this README file) describing your problem formulation, proposed approach, each team member's responsibilities, proposed baseline models with Hugging Face links, proposed evaluation metrics, proposed data with links, and goals for each week.

### 1. Problem formulation and social impact
- **Problem:** Dandelion Africa field/community health workers report on their activities (home visits, trainings, livelihood distributions, patient follow-ups) informally in WhatsApp group chats, mixing text and voice notes with unrelated chit-chat. Program/M&E staff currently have to read through this manually to know what happened. We will build a pipeline that converts the WhatsApp export (text + audio) into structured records of field activities and outcomes: filtering out irrelevant messages, transcribing audio, segmenting conversation into discrete events, extracting structured attributes, deduplicating repeated reports of the same event, and estimating confidence so uncertain extractions get routed to a human for verification.
- **Intended users:** Dandelion Africa program/M&E (monitoring & evaluation) staff who need a structured activity log instead of scrolling through raw chat history.
- **Inputs and outputs:** Input = a WhatsApp chat export (ordered messages with sender, timestamp, text and/or voice-note audio). Output = a list of structured event records. Example: input message "Visited Mama A.'s homestead today, gave 2 goats as part of the livelihood program, she was very happy" (voice note, transcribed) → output `{event_type: "livelihood_distribution", date: "2026-02-03", beneficiary: "Mama A. (anonymized)", quantity: "2 goats", outcome: "positive", confidence: 0.82, source_msg_id: 145}`.
- **Success criteria:** The pipeline correctly identifies which messages describe real activities (high precision/recall), extracts the right structured fields for those activities, never leaks PII into the output, and flags low-confidence or ambiguous extractions for human review rather than guessing.
- **Constraints and risks:** No data available; ASR errors on accented speech will cascade into wrong extractions; LLM-based extraction can hallucinate plausible-but-unsupported details; anonymization must have very high recall since a missed name/phone number is a real privacy failure, not just a quality issue; synthetic data will not fully represent the real distribution of messages.

### 2. Proposed technical approach
- **Pipeline:**
  1. **Parse** — parse the WhatsApp export (chat log + audio attachments).
  2. **ASR** — transcribe voice notes to text.
  3. **Anonymization** — detect and redact PII (names, ages, locations, phone numbers) in both text and transcripts.
  4. **Relevance filtering** — classify each message as a real field-activity report vs. irrelevant chit-chat.
  5. **Event segmentation** — group related messages (same thread/time window) into one activity "event."
  6. **Attribute extraction** — pull structured fields (activity type, date, location, quantity, outcome) from each event, schema-constrained to reduce hallucination.
  7. **Deduplication** — detect multiple messages reporting the same underlying event.
  8. **Confidence estimation** — combine ASR confidence, extraction self-consistency, and rule checks into a per-record confidence score.
  9. **Human verification queue** — route low-confidence/ambiguous records for review (exported as a simple spreadsheet).
- **Baseline(s):** Simplest end-to-end version: keyword/regex relevance filter + Whisper (small) for ASR + a single few-shot LLM prompt that does filtering, anonymization, and structured JSON extraction in one pass, with no dedup or confidence scoring yet. This gets raw input → structured output working end-to-end, which we then improve module by module.
- **Proposed models/tools:**
  - [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) — ASR for voice notes; large-v3 chosen over small/base for better robustness to accented English (fallback to `whisper-small` if latency/compute is an issue).
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) — speaker diarization if a voice note or export segment has multiple speakers.
  - [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) — NER-based detection of names/locations for anonymization (combined with regex for phone numbers/ages); Microsoft Presidio evaluated as an alternative/complementary toolkit.
  - [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) — zero-shot classifier for the relevance-filtering step before we have enough labeled data to fine-tune.
  - [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — sentence embeddings for similarity-based deduplication of repeated reports.
  - [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) — instruction-following LLM for schema-constrained structured extraction.
- **Alternative approach:** If ASR accuracy on accented speech is too low, fine-tune/adapt Whisper on accented-English speech data (e.g., [AfriSpeech-200](https://huggingface.co/datasets/intronhealth/afrispeech-200)) instead of using it zero-shot. If LLM extraction hallucinates too often, move from free-form prompting to grammar/schema-constrained decoding (e.g., Outlines or a JSON-schema-enforced generation library) or a smaller fine-tuned extraction model.

### 3. Proposed data
| Dataset/source | Link or access instructions | Purpose | Size / languages | Privacy or usage considerations |
|---|---|---|---|---|
| Team-authored synthetic WhatsApp text | Created by team (scripts in `data/synthetic/`) | Training/development/evaluation | ~50–100 synthetic messages/threads to start, English, mix of relevant activity reports + chit-chat | All names/phone numbers/locations are fabricated; still passed through the anonymization step before being used as examples |
| Team-recorded synthetic voice notes | Recorded by team members/friends/family (or TTS fallback) | Development/evaluation of ASR + full pipeline | Target: several accents/voices, majority women speakers, English | Only volunteers who consent to a synthetic, fabricated script; no real personal disclosures recorded |
| [AfriSpeech-200](https://huggingface.co/datasets/intronhealth/afrispeech-200) | Hugging Face dataset | ASR robustness reference/validation (African-accented English) | ~200 hrs, African-accented English | Public research dataset; used only as an ASR benchmark, not Dandelion-specific content |
| [Mozilla Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Hugging Face dataset | Supplementary ASR validation, accent diversity | Multilingual, accent-tagged subset usable | Public, consented crowd-sourced dataset |

- **Data preparation:** Write a small generator/script for WhatsApp-export-style synthetic threads (sender, timestamp, text/audio); record or TTS-generate the corresponding audio; manually label a gold dev set (relevant/irrelevant, event type + attributes, injected fake PII) for evaluation; keep raw synthetic recordings out of version control, commit only anonymized derived examples.
- **Data limitations:** Synthetic data will not capture real code-switching (Swahili/Dholuo mixed with English), background noise, or the true activity taxonomy Dandelion Africa actually uses, we are approximating based on the challenge description until we can validate against real program vocabulary.

### 4. Proposed evaluation
| Metric / test | What it measures | How it will be calculated | Target / success criterion |
|---|---|---|---|
| Relevance F1 | Quality of filtering real activity reports from chit-chat | Precision/recall/F1 vs. gold-labeled dev messages | F1 ≥ 0.8 on dev set |
| Word Error Rate (WER) | ASR transcription quality | WER of Whisper transcript vs. gold transcript on recorded dev audio | WER low enough that extraction errors are not ASR-dominated (target ≤ 20% as a first bar) |
| Field-level extraction accuracy | Correctness of structured attributes extracted per event | Per-slot exact/partial match vs. gold structured record | ≥ 0.75 per-slot accuracy on dev set |
| PII recall (anonymization) | Whether all injected PII is caught | % of injected fake names/ages/locations/phone numbers successfully redacted | As close to 100% as possible — this is a safety-critical metric, not just a quality one |
| Deduplication precision/recall | Whether repeated reports of the same event are correctly merged | Pairwise match vs. gold duplicate-pairs labels | To be set once we have enough duplicate examples in the dev set |
| Confidence calibration | Whether low-confidence flags correlate with actual errors | AUROC of confidence score vs. correctness on dev set | Confidence score meaningfully separates correct from incorrect extractions |

- **Evaluation examples:** We will build our own labeled dev set (~30–50 synthetic threads, mixing relevant/irrelevant messages, injected PII, and some duplicate reports) and have each team member independently label a subset, then reconcile disagreements to produce the gold set.
- **Human verification:** Team members cross-check each other's gold labels before finalizing; low-confidence pipeline outputs are routed to a reviewable queue rather than accepted automatically.
- **Failure cases to watch:** (1) ASR mishearing accented speech and causing the wrong activity type/quantity to be extracted; (2) a misspelled or ASR-garbled name slipping past the anonymization NER model; (3) the extraction LLM inventing a plausible but unstated detail (e.g., a quantity or date not actually present in the message).

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
- **What runs end-to-end now?** [Describe exactly.]
- **How to run it:** [Commands, dependencies, configuration; refer to Setup below.]
- **Model(s) and data actually used:** [Names, versions, links, and any changes from the plan.]
- **Example 1:** Input: [non-sensitive example] → Output: [result]
- **Example 2:** Input: [non-sensitive example] → Output: [result]

### Attempts and outcomes
| Attempt / change | Worked? (Yes / Partly / No) | Evidence (output, metric, link) | What we learned / next step |
|---|---|---|---|
| [Attempt] | [Status] | [Evidence] | [Learning] |

### Individual contributions during that week
| Member | What I personally implemented/tested | Result (worked / partly / failed) | Evidence (commit, PR, experiment) | Blocker / next step |
|---|---|---|---|---|
| [Name] | [Specific contribution] | [Status] | [Link] | [Next step] |

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
[Python version, libraries, compute requirements, model access]

### Installation
```bash
# Add installation commands here
```

### Run the system
```bash
# Add an example command here
```

### Run evaluation
```bash
# Add an example evaluation command here
```

### Repository structure
```text
project/
├── README.md
├── requirements.txt
├── src/
├── tests/
├── evaluation/
└── examples/
```


