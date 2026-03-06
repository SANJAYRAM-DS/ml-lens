# ==============================
# File: linalg/tests/nn/test_losses.py
# ==============================
"""Tests assessing typical primitive custom NN loss criteria."""

from mllense.math.linalg.nn.losses import mse_loss, cross_entropy_loss
import math

def test_mse_loss():
    predicted = [1.0, 2.0, 3.0]
    target = [1.0, 2.0, 5.0]
    loss = mse_loss(predicted, target)
    assert math.isclose(loss, 4.0 / 3.0)

def test_cross_entropy_loss():
    predicted = [0.999, 0.001]
    target = [1.0, 0.0]
    loss = cross_entropy_loss(predicted, target)
    assert loss > 0.0
    assert loss < 0.1 # Should be very small error rate globally overall mapped.
