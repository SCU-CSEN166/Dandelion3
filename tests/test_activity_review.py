import copy
import unittest
from unittest.mock import patch

from src.activity_review import process_messages


def sample_event():
    return {
        "event_id": "evt_0042",
        "source_message_ids": [3],
        "text": "Gave 2 goats to Amina in Kakamega.",
        "relevance": "event",
        "relevance_confidence": 0.94,
        "activity_type": "livelihood distribution",
        "participants": 1,
        "duplicate_of": "evt_0001",
        "duplicate_confidence": 0.81,
    }


class ActivityReviewTests(unittest.TestCase):
    def setUp(self):
        # Model behavior is tested separately; these tests check the handoff.
        patcher = patch("src.activity_review.anonymize_text", side_effect=lambda text, model: text)
        self.anonymize = patcher.start()
        self.addCleanup(patcher.stop)

    def test_preserves_event_id_and_duplicate_details(self):
        event = process_messages([sample_event()])[0]
        self.assertEqual(event["event_id"], "evt_0042")
        self.assertEqual(event["duplicate_of"], "evt_0001")
        self.assertEqual(event["duplicate_confidence"], 0.81)

    def test_only_new_fields_are_required(self):
        event = process_messages([sample_event()])[0]
        self.assertEqual(event["attributes"], {
            "activity_type": "livelihood distribution", "participants": 1,
        })
        self.assertEqual(event["confidence"]["score"], 0.94)
        self.assertFalse(event["confidence"]["needs_review"])

    def test_raw_text_and_activity_are_anonymized(self):
        self.anonymize.side_effect = lambda text, model: (
            text.replace("Amina", "[PERSON]").replace("Kakamega", "[LOCATION]")
        )
        message = sample_event()
        message["activity_type"] = "training in Kakamega"
        original = copy.deepcopy(message)
        event = process_messages([message])[0]
        self.assertEqual(event["anonymized_text"], "Gave 2 goats to [PERSON] in [LOCATION].")
        self.assertEqual(event["attributes"]["activity_type"], "training in [LOCATION]")
        self.assertNotIn("text", event)
        self.assertEqual(message, original)

    def test_cluster_output_can_be_reviewed(self):
        from src.cluster import cluster_events

        events = cluster_events([{
            "id": 3, "text": "Forty-two women attended the training.",
            "activity_type": "training", "participants": 42,
            "relevance_confidence": 0.94,
        }])
        events[0].update(event_id="evt_0001", relevance="event", duplicate_of=None)
        reviewed = process_messages(events)[0]
        self.assertEqual(reviewed["source_message_ids"], [3])
        self.assertEqual(reviewed["attributes"]["participants"], 42)
        self.assertFalse(reviewed["confidence"]["needs_review"])

    def test_chitchat_is_skipped(self):
        self.assertEqual(process_messages([{"relevance": "chitchat"}]), [])
        self.anonymize.assert_not_called()

    def test_participant_count_is_preserved(self):
        for count in (0, 42):
            with self.subTest(count=count):
                message = sample_event()
                message["participants"] = count
                event = process_messages([message])[0]
                self.assertEqual(event["attributes"]["participants"], count)
                self.assertFalse(event["confidence"]["needs_review"])

    def test_invalid_participant_count_is_rejected(self):
        for count in (-1, True, 2.5, "42"):
            with self.subTest(count=count):
                message = sample_event()
                message["participants"] = count
                with self.assertRaisesRegex(ValueError, "participants"):
                    process_messages([message])

    def test_missing_participants_requires_review(self):
        message = sample_event()
        del message["participants"]
        confidence = process_messages([message])[0]["confidence"]
        self.assertEqual(confidence["score"], 0.6)
        self.assertTrue(confidence["needs_review"])
        self.assertIn("Missing fields: participants", confidence["reasons"])

    def test_conflicting_participants_are_not_added_together(self):
        first = sample_event()
        second = copy.deepcopy(first)
        second["source_message_ids"] = [4]
        first["participants"] = 42
        second["participants"] = 40
        event = process_messages([first, second])[0]
        self.assertIsNone(event["attributes"]["participants"])
        self.assertEqual(event["conflicts"]["participants"], [42, 40])
        self.assertTrue(event["confidence"]["needs_review"])

    def test_conflicts_still_require_review(self):
        first = sample_event()
        first.pop("event_id")
        first["activity_key"] = "visit_1"
        second = copy.deepcopy(first)
        second["source_message_ids"] = [4]
        second["activity_type"] = "training"
        event = process_messages([first, second])[0]
        self.assertIsNone(event["attributes"]["activity_type"])
        self.assertEqual(event["conflicts"]["activity_type"], ["livelihood distribution", "training"])
        self.assertTrue(event["confidence"]["needs_review"])
        self.assertEqual(event["confidence"]["score"], 0.4)

    def test_low_relevance_still_requires_review(self):
        message = sample_event()
        message["relevance_confidence"] = 0.55
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
