"""
Full three-person chain: Sathvik's ingestion -> Roland's mid-pipeline ->
Ronav's activity_baseline review layer. Regression test for a real
interface mismatch found while testing the whole system together:
activity_baseline.py expects a bare list of event records, but
pipeline.py's main output is a dict with `events`/`irrelevant_messages`
keys. Fixed by having pipeline.py also write a bare events list
(`--events-output`); this test checks that handoff actually works end to
end, not just that each piece works alone.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_full_chain_ingestion_to_review(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    from activity_baseline import process_messages
    from pipeline import run_pipeline

    output = run_pipeline(ROOT / "data/synthetic/whatsapp_export.txt")

    # This is the exact handoff shape Ronav's script expects: a bare list,
    # not the {"events": ..., "irrelevant_messages": ...} wrapper.
    reviewed = process_messages(output["events"])

    assert len(reviewed) == len(output["events"]), "the review layer shouldn't silently drop events"
    for event in reviewed:
        assert "confidence" in event and "needs_review" in event["confidence"]
