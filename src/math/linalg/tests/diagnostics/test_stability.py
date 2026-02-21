# ==============================
# File: linalg/tests/diagnostics/test_stability.py
# ==============================
"""Tests for stability reporting routines."""

from mllense.math.linalg.diagnostics.stability import stability_report

def test_stability_report_good():
    a = [[1.0, 0.0], [0.0, 1.0]]
    report = stability_report(a)
    assert report["rank"] == 2
    assert report["full_rank"] is True
    assert report["well_conditioned"] is True

def test_stability_report_singular():
    a = [[1.0, 2.0], [2.0, 4.0]]
    report = stability_report(a)
    assert report["rank"] == 1
    assert report["full_rank"] is False
    assert report["well_conditioned"] is False
