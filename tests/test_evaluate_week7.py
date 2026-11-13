"""Regression test for the Week 7 evaluation script: makes sure every
behavioral check actually finds its event (no silent 'event not found'),
and that the known pass/fail split hasn't silently drifted."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_week7_runs_clean():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/evaluate_week7.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "event not found" not in result.stdout, (
        "a behavioral check didn't find its event in the current data -- "
        "check scripts/evaluate_week7.py's search substrings"
    )
    assert "2/11 passed" in result.stdout, (
        "the known pass/fail split (2/11) changed -- re-verify the new result "
        "is real before updating this test's expectation"
    )
