import copy
import unittest

from src.activity_baseline import process_messages


def sample_event():
    return {
        "event_id": "evt_0042",
        "source_message_ids": [3],
        "anonymized_text": "Gave 2 goats to [PERSON] in Kakamega.",
        "relevance": {"label": "relevant", "confidence": 0.94},
        "attributes": {
            "activity_type": "livelihood_distribution",
            "date": "2026-02-03",
            "location": "Kakamega",
            "quantity": "2 goats",
            "outcome": "positive",
        },
        "duplicate_of": "evt_0001",
        "duplicate_confidence": 0.81,
    }


class ActivityBaselineTests(unittest.TestCase):
    def test_preserves_event_id_and_duplicate_details(self):
        event = process_messages([sample_event()])[0]
        self.assertEqual(event["event_id"], "evt_0042")
        self.assertEqual(event["duplicate_of"], "evt_0001")
        self.assertEqual(event["duplicate_confidence"], 0.81)

    def test_missing_location_only_suggests_review(self):
        message = sample_event()
        message["attributes"]["location"] = None
        confidence = process_messages([message])[0]["confidence"]
        self.assertEqual(confidence["score"], 0.94)
        self.assertFalse(confidence["needs_review"])
        self.assertTrue(confidence["suggested_review"])
        self.assertEqual(confidence["suggestions"], ["Missing location"])

    def test_missing_quantity_still_requires_review(self):
        message = sample_event()
        message["attributes"]["location"] = None
        message["attributes"]["quantity"] = None
        confidence = process_messages([message])[0]["confidence"]
        self.assertEqual(confidence["score"], 0.6)
        self.assertTrue(confidence["needs_review"])
        self.assertTrue(confidence["suggested_review"])

    def test_conflicts_still_require_review(self):
        first = sample_event()
        first.pop("event_id")
        first["activity_key"] = "visit_1"
        second = copy.deepcopy(first)
        second["source_message_ids"] = [4]
        second["attributes"]["quantity"] = "3 goats"
        event = process_messages([first, second])[0]
        self.assertIsNone(event["attributes"]["quantity"])
        self.assertEqual(event["conflicts"]["quantity"], ["2 goats", "3 goats"])
        self.assertTrue(event["confidence"]["needs_review"])
        self.assertEqual(event["confidence"]["score"], 0.4)

    def test_low_relevance_still_requires_review(self):
        message = sample_event()
        message["attributes"]["location"] = None
        message["relevance"]["confidence"] = 0.55
        confidence = process_messages([message])[0]["confidence"]
        self.assertEqual(confidence["score"], 0.55)
        self.assertTrue(confidence["needs_review"])

    def test_distinct_existing_events_keep_their_ids(self):
        first = sample_event()
        first["activity_key"] = "visit_1"
        second = copy.deepcopy(first)
        second["event_id"] = "evt_0001"
        second["source_message_ids"] = [4]
        events = process_messages([first, second, copy.deepcopy(first)])
        self.assertEqual([event["event_id"] for event in events], ["evt_0042", "evt_0001"])

    def test_generated_ids_do_not_replace_existing_ids(self):
        first = sample_event()
        first.pop("event_id")
        second = sample_event()
        second["event_id"] = "evt_0001"
        second["source_message_ids"] = [4]
        events = process_messages([first, second])
        self.assertEqual([event["event_id"] for event in events], ["evt_0002", "evt_0001"])


if __name__ == "__main__":
    unittest.main()
