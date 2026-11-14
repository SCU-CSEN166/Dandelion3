"""
Week 7 class demo: runs the real system live and prints a presentation-ready
summary of metrics plus the representative success/failure cases from
TESTING.md -- computed from the actual current run, not hardcoded numbers.

Usage:
    python scripts/demo_week7.py

Models download on first run (facebook/bart-large-mnli, dslim/bert-base-NER,
sentence-transformers/all-MiniLM-L6-v2) -- run it once before the actual
presentation so it's fast live.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from activity_review import process_messages
from pipeline import run_pipeline
from relevance_classifier import classify_relevance, evaluate, parse_whatsapp_export

INPUT = ROOT / "data/synthetic/whatsapp_export.txt"
GOLD = ROOT / "data/synthetic/gold_labels.jsonl"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def find_event(events, *substrings):
    """First event whose anonymized_text contains all given substrings."""
    for event in events:
        text = event["anonymized_text"]
        if all(s in text for s in substrings):
            return event
    return None


def main() -> None:
    section("1. DATASET")
    messages = parse_whatsapp_export(INPUT)
    n_relevant_gold = sum(1 for line in GOLD.read_text().splitlines() if '"relevant"' in line)
    print(f"{len(messages)} synthetic WhatsApp messages ({INPUT.name})")
    print(f"  {n_relevant_gold} relevant / {len(messages) - n_relevant_gold} irrelevant (gold labels)")
    print("  26 are the original dev set; 11 (messages 27-37) are deliberately")
    print("  adversarial, written to probe each pipeline step's failure modes.")

    section("2. BASELINE (Week 6): relevance classifier alone")
    predictions = classify_relevance(messages, "facebook/bart-large-mnli")
    metrics = evaluate(predictions, GOLD)
    print(f"Relevance F1 on the current {len(messages)}-message dataset: {metrics}")
    print("(On the original 24-message Week 6 dataset, before the adversarial cases")
    print(" were added, this was F1 0.957 -- see TESTING.md for that run.)")

    section("3. CURRENT (Week 7): full chain")
    print("ingestion -> anonymize -> relevance filter -> segment -> extract -> dedup")
    print("-> Ronav's review layer\n")
    output = run_pipeline(INPUT)
    events = output["events"]
    n_dupes = sum(1 for e in events if e["duplicate_of"])
    print(f"-> {len(events)} events ({n_dupes} flagged as duplicates), "
          f"{len(output['irrelevant_messages'])} filtered out as irrelevant")

    reviewed = process_messages(events)
    n_flagged = sum(1 for e in reviewed if e["confidence"]["needs_review"])
    print(f"-> chained into activity_review.py: {n_flagged}/{len(reviewed)} "
          f"({n_flagged / len(reviewed):.0%}) flagged needs_review")

    section("4. SUCCESS CASE: segmentation + deduplication working together")
    goat_event = find_event(events, "homestead", "2 goats", "pamphlets")
    if goat_event:
        print(f"Event {goat_event['event_id']} merged source messages "
              f"{goat_event['source_message_ids']} (same sender, 2 minutes apart):")
        print(f"  \"{goat_event['anonymized_text']}\"")
        print(f"  -> attributes: {goat_event['attributes']}")
        dup = find_event(events, "confirming", "2 goats", "pleased")
        if dup:
            print(f"\nA later, independently-worded report of the SAME event "
                  f"(message {dup['source_message_ids']}, different sender):")
            print(f"  \"{dup['anonymized_text']}\"")
            print(f"  -> correctly flagged duplicate_of: {dup['duplicate_of']} "
                  f"(similarity {dup['duplicate_confidence']})")
    else:
        print("(case not found in current data -- see TESTING.md for the recorded result)")

    section("5. FAILURE CASE: anonymization vs. deduplication tension")
    false_dup = find_event(events, "thrilled")  # "Oduya" itself gets redacted by anonymization
    if false_dup:
        print(f"A genuinely DIFFERENT beneficiary's event:")
        print(f"  \"{false_dup['anonymized_text']}\"")
        print(f"  -> wrongly flagged duplicate_of: {false_dup['duplicate_of']} "
              f"(similarity {false_dup['duplicate_confidence']})")
        print("  Why: anonymization redacted the one detail (the beneficiary's name)")
        print("  that would have told these two real events apart.")
    else:
        print("(case not found in current data -- see TESTING.md for the recorded result)")

    section("6. FAILURE CASE: segmentation has no topic awareness")
    merged_wrong = find_event(events, "mosquito nets", "composting")
    if merged_wrong:
        print(f"Event {merged_wrong['event_id']} wrongly merged two DIFFERENT activities")
        print(f"(same sender, 3 minutes apart):")
        print(f"  \"{merged_wrong['anonymized_text']}\"")
        print(f"  -> attributes only reflect the second activity: {merged_wrong['attributes']}")
        print("  The mosquito-net distribution's details are lost entirely.")
    else:
        print("(case not found in current data -- see TESTING.md for the recorded result)")

    section("7. BUGS FOUND BY RUNNING THE REAL SYSTEM (not just unit tests)")
    print("- Anonymization corrupted text when one name split into multiple NER")
    print("  entities (\"Otieno\" -> \"[PERSON]N]N]\") -- fixed, regression test added.")
    print("- Sathvik's parser defaulted to month-first dates with no ambiguous date")
    print("  to detect the convention from -- fixed with an opt-in dayfirst override.")
    print("- Ronav's review layer crashed on our real output (expects a bare list,")
    print("  we returned a dict) -- fixed by also writing the bare-list shape.")
    print("\nFull details, every threshold, and every test: see TESTING.md")


if __name__ == "__main__":
    main()
