# Synthetic data

`whatsapp_export.txt` is a fabricated WhatsApp-export-style chat log written by the team
(no real Dandelion Africa data — see the top-level README for why). 37 messages, mixing:

- **Relevant** messages: made-up field-activity reports (home visits, trainings,
  distributions, follow-ups), including fabricated beneficiary names, ages, locations,
  and a phone number, used to test the anonymization step.
- **Irrelevant** messages: casual group chit-chat.
- Messages 3+4: a two-message report of the same event from the same sender, two minutes
  apart, to test event segmentation.
- Message 26: an independently-worded report of the *same* event as message 3+4, from a
  different sender much later in the chat, to test deduplication.
- **Messages 27–37**: deliberately adversarial cases added to find where the baseline
  breaks (vague relevance, a phone number format our regex misses, two activities merged
  by segmentation, one activity wrongly split by segmentation, an uncategorized activity,
  a word-form quantity, an unflagged negative outcome, a heavily-paraphrased duplicate, and
  a different beneficiary wrongly flagged as a duplicate). See [`TESTING.md`](../../TESTING.md)
  for exactly what happened with each one.

`gold_labels.jsonl` has the true `relevant` / `irrelevant` label for each message, by `id`
(1-indexed, in file order), for evaluating the relevance classifier.

`sample_messages.json` is Sathvik's ingestion-format sample data (a different, simpler
schema used to test the parser in `src/parse_messages.py`).
