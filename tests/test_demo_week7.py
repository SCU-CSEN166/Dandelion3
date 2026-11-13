"""Regression test for the Week 7 demo script: makes sure every section
actually finds its case in the current data, rather than silently printing
the "case not found" fallback during a live demo."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_demo_finds_every_case():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/demo_week7.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert "case not found" not in result.stdout, (
        "a demo case didn't match the current data -- check scripts/demo_week7.py's "
        "search substrings against the real pipeline output"
    )
