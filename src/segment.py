"""
Event segmentation (pipeline step 5): group consecutive relevant messages
into one activity "event".

Baseline rule: two relevant messages merge into one event if they're from
the same sender and at most `max_gap_minutes` apart (gaps are measured
between consecutive *relevant* messages, so intervening chit-chat doesn't
break a real event thread). This has no notion of topic, just sender +
time, which is a known limitation — see README.
"""

from datetime import datetime


def _parse_dt(date: str, time: str) -> datetime:
    return datetime.strptime(f"{date} {time}", "%d/%m/%Y %H:%M")


def segment_events(relevant_messages: list[dict], max_gap_minutes: int = 10) -> list[dict]:
    events: list[dict] = []

    for msg in relevant_messages:
        if events:
            last = events[-1]
            same_sender = last["sender"] == msg["sender"]
            gap_minutes = (
                _parse_dt(msg["date"], msg["time"]) - _parse_dt(last["_last_date"], last["_last_time"])
            ).total_seconds() / 60
            if same_sender and 0 <= gap_minutes <= max_gap_minutes:
                last["source_message_ids"].append(msg["id"])
                last["anonymized_text"] += " " + msg["anonymized_text"]
                last["_last_date"] = msg["date"]
                last["_last_time"] = msg["time"]
                continue

        events.append(
            {
                "source_message_ids": [msg["id"]],
                "sender": msg["sender"],
                "anonymized_text": msg["anonymized_text"],
                "_last_date": msg["date"],
                "_last_time": msg["time"],
            }
        )

    for i, event in enumerate(events, start=1):
        event["event_id"] = f"evt_{i:04d}"
        del event["_last_date"]
        del event["_last_time"]

    return events
