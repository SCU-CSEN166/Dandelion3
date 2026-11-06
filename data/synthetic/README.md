# Synthetic data

`whatsapp_export.txt` is a fabricated WhatsApp-export-style chat log written by the team
(no real Dandelion Africa data — see the top-level README for why). It mixes:

- **Relevant** messages: made-up field-activity reports (home visits, trainings,
  distributions, follow-ups), including fabricated beneficiary names, ages, locations,
  and a phone number, used to later test the anonymization step.
- **Irrelevant** messages: casual group chit-chat.

`gold_labels.jsonl` has the true `relevant` / `irrelevant` label for each message, by `id`
(1-indexed, in file order), for evaluating the relevance classifier.
