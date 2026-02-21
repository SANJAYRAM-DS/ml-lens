# ==============================
# File: linalg/tests/core/test_trace.py
# ==============================
"""Tests for trace system."""

from mllense.math.linalg.core.trace import Trace

def test_trace_enabled():
    trace = Trace(enabled=True)
    trace.record(operation="test_op", description="testing...", data={"status": "ok"})
    
    events = trace.steps
    assert len(events) == 1
    assert events[0].operation == "test_op"
    assert events[0].description == "testing..."
    assert events[0].data == {"status": "ok"}
    assert "status" in trace.replay()
    
def test_trace_disabled():
    trace = Trace(enabled=False)
    trace.record("test_op", "testing...")
    
    events = trace.steps
    assert len(events) == 0
    assert trace.replay() == "(no trace recorded)"
