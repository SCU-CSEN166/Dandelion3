"""
Week 7 class demo: real, comprehensive evaluation -- runs the system,
computes pass/fail for every behavioral check we have a defined correct
answer for, prints the numeric rates FIRST, then every single result
(not a curated subset). Nothing here is narrated; every PASS/FAIL below
is computed from the actual pipeline output of this run.

Usage:
    python scripts/evaluate_week7.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import run_pipeline
from relevance_classifier import classify_relevance, evaluate, parse_whatsapp_export

INPUT = ROOT / "data/synthetic/whatsapp_export.txt"
GOLD = ROOT / "data/synthetic/gold_labels.jsonl"


def find_event(events, *substrings):
    for event in events:
        text = event["anonymized_text"]
        if all(s in text for s in substrings):
            return event
    return None


def main() -> None:
    # ---------------------------------------------------------------
    # PART 1: Relevance filtering -- real precision/recall/F1
    # ---------------------------------------------------------------
    messages = parse_whatsapp_export(INPUT)
    predictions = classify_relevance(messages, "facebook/bart-large-mnli")
    metrics = evaluate(predictions, GOLD)

    print("=" * 72)
    print(f"RELEVANCE FILTERING -- {len(messages)} messages")
    print("=" * 72)
    print(f"Precision: {metrics['precision']}   Recall: {metrics['recall']}   "
          f"F1: {metrics['f1']}")
    print(f"TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")

    gold = {}
    for line in GOLD.read_text().splitlines():
        import json
        row = json.loads(line)
        gold[row["id"]] = row["label"]

    print("\nMisclassified messages (every one, not a sample):")
    n_errors = 0
    for pred in predictions:
        truth = gold.get(pred["id"])
        if truth is None or truth == pred["predicted_label"]:
            continue
        n_errors += 1
        kind = "FALSE POSITIVE (predicted relevant, actually irrelevant)" \
            if pred["predicted_label"] == "relevant" else \
            "FALSE NEGATIVE (predicted irrelevant, actually relevant)"
        print(f"  [{kind}] id={pred['id']} confidence={pred['confidence']}")
        print(f"    \"{pred['text']}\"")
    if n_errors == 0:
        print("  (none)")

    # ---------------------------------------------------------------
    # PART 2: Pipeline behavioral checks -- every one we have ground
    # truth for, not a curated subset.
    # ---------------------------------------------------------------
    output = run_pipeline(INPUT)
    events = output["events"]

    checks = []

    def add_check(name, expected, actual, passed):
        checks.append({"name": name, "expected": expected, "actual": actual, "passed": passed})

    e = find_event(events, "homestead", "pamphlets")
    add_check(
        "Segmentation: 2-message same-sender thread should merge",
        "2 messages merged into 1 event",
        f"{len(e['source_message_ids'])} message(s) merged into 1 event" if e else "event not found",
        bool(e and len(e["source_message_ids"]) == 2),
    )

    e = find_event(events, "confirming", "pleased")
    add_check(
        "Dedup: later re-report of same event should be flagged duplicate",
        "duplicate_of set", f"duplicate_of={e['duplicate_of']}" if e else "event not found",
        bool(e and e["duplicate_of"]),
    )

    e = find_event(events, "mosquito nets", "composting")
    add_check(
        "Segmentation: two DIFFERENT activities, same sender 3min apart, should NOT merge",
        "stay as 2 separate events", "merged into 1 event" if e else "correctly stayed separate",
        e is None,
    )

    e = find_event(events, "goat shed construction", "roofing")
    add_check(
        "Segmentation: real continuation 12min apart SHOULD merge (currently a known gap)",
        "merge into 1 event", "merged" if e else "stayed split into 2 events",
        e is not None,
    )

    e = find_event(events, "borehole pump")
    add_check(
        "Extraction: real but unlisted activity type should not fall back to 'other'",
        "activity_type != 'other'", f"activity_type={e['attributes']['activity_type']}" if e else "event not found",
        bool(e and e["attributes"]["activity_type"] != "other"),
    )

    e = find_event(events, "blankets")
    add_check(
        "Extraction: word-form quantity ('twenty two') should be extracted",
        "quantity is not None", f"quantity={e['attributes']['quantity']}" if e else "event not found",
        bool(e and e["attributes"]["quantity"] is not None),
    )

    e = find_event(events, "refused the proposed support")
    add_check(
        "Extraction: negative/refusal outcome should not read as 'unknown'",
        "outcome != 'unknown'", f"outcome={e['attributes']['outcome']}" if e else "event not found",
        bool(e and e["attributes"]["outcome"] != "unknown"),
    )

    e = find_event(events, "follow-up visit")
    phone_leaked = bool(e and "0712" in e["anonymized_text"])
    add_check(
        "Anonymization: dot-separated phone number should be redacted",
        "'0712...' not in text", "phone number still present" if phone_leaked else "redacted",
        not phone_leaked,
    )
    bogus_qty = bool(e and e["attributes"]["quantity"] and "678" in e["attributes"]["quantity"])
    add_check(
        "Extraction: should not extract garbage quantity from an un-redacted phone number",
        "quantity doesn't contain stray digits from the phone number",
        f"quantity={e['attributes']['quantity']}" if e else "event not found",
        not bogus_qty,
    )

    e = find_event(events, "Circling back", "handover")
    add_check(
        "Dedup: heavily-paraphrased true duplicate SHOULD be flagged (currently a known gap)",
        "duplicate_of set", f"duplicate_of={e['duplicate_of']}" if e else "event not found",
        bool(e and e["duplicate_of"]),
    )

    e = find_event(events, "thrilled")
    add_check(
        "Dedup: a DIFFERENT real beneficiary should NOT be flagged as a duplicate",
        "duplicate_of is None", f"duplicate_of={e['duplicate_of']}" if e else "event not found",
        bool(e and e["duplicate_of"] is None),
    )

    n_passed = sum(1 for c in checks if c["passed"])
    print("\n" + "=" * 72)
    print(f"PIPELINE BEHAVIORAL CHECKS -- {n_passed}/{len(checks)} passed "
          f"({n_passed / len(checks):.0%})")
    print("=" * 72)
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"\n[{status}] {c['name']}")
        print(f"  expected: {c['expected']}")
        print(f"  actual:   {c['actual']}")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Relevance filtering F1: {metrics['f1']} ({len(messages) - n_errors}/{len(messages)} correct)")
    print(f"Pipeline behavioral checks: {n_passed}/{len(checks)} passed ({n_passed / len(checks):.0%})")


if __name__ == "__main__":
    main()
