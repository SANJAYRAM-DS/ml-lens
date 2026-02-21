# ==============================
# File: linalg/tests/nn/test_activations.py
# ==============================
"""Tests for activation primitive components."""

from mllense.math.linalg.nn.activations import relu, sigmoid, tanh, softmax
import math

def test_relu():
    res = relu([-1.0, 0.0, 2.0])
    assert res == [0.0, 0.0, 2.0]
    
    res_mat = relu([[-1.0, 2.5], [0.0, -0.5]])
    assert res_mat == [[0.0, 2.5], [0.0, 0.0]]

def test_sigmoid():
    res = sigmoid([0.0])
    assert math.isclose(res[0], 0.5)

def test_softmax():
    res = softmax([0.0, 0.0, 0.0])
    assert math.isclose(sum(res), 1.0)
    assert math.isclose(res[0], 1.0 / 3.0)
    assert math.isclose(res[1], 1.0 / 3.0)
