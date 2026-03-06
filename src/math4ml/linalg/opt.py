import numpy as np

def grad_matmul(dL_dC, A, B, help=False, what=False):
    dL_dA = dL_dC @ B.T
    dL_dB = A.T @ dL_dC
    explanation = ""

    if what:
        explanation += "WHAT: Computes how the loss changes with A and B when C = A @ B.\n"
        explanation += "Formula:\n"
        explanation += "  dL/dA = dL/dC @ B^T\n"
        explanation += "  dL/dB = A^T @ dL/dC\n"
        explanation += f"Example result: dA={dL_dA}, dB={dL_dB}\n"
        return dL_dA, dL_dB, explanation

    if help:
        explanation += "Step 1: Compute gradient w.r.t A using dL/dA = dL/dC @ B^T\n"
        explanation += f"dL/dC:\n{dL_dC}\nB^T:\n{B.T}\n"
        explanation += f"dL/dA =\n{dL_dA}\n\n"
        explanation += "Step 2: Compute gradient w.r.t B using dL/dB = A^T @ dL/dC\n"
        explanation += f"A^T:\n{A.T}\n"
        explanation += f"dL/dB =\n{dL_dB}\n"

    return dL_dA, dL_dB, explanation


def grad_relu(x, help=False, what=False):
    grad = (x > 0).astype(x.dtype)
    explanation = ""

    if what:
        explanation += "WHAT: ReLU gradient tells which values pass the derivative.\n"
        explanation += "If x > 0 → gradient = 1\nIf x ≤ 0 → gradient = 0\n"
        explanation += f"Example: grad_relu({x}) = {grad}\n"
        return grad, explanation

    if help:
        explanation += "ReLU gradient:\n"
        explanation += "For each element: positive → 1, else → 0\n"
        explanation += f"Input: {x}\nGradient: {grad}\n"

    return grad, explanation


def sigmoid(x, help=False, what=False):
    s = 1 / (1 + np.exp(-x))
    explanation = ""

    if what:
        explanation += "WHAT: Sigmoid squashes numbers into (0,1). Formula:\n"
        explanation += "  sigmoid(x) = 1 / (1 + exp(-x))\n"
        explanation += f"Example: sigmoid({x}) = {s}\n"
        return s, explanation

    if help:
        explanation += "Sigmoid activation:\n"
        explanation += "Compute 1 / (1 + exp(-x)) for each element\n"
        explanation += f"Input: {x}\nOutput: {s}\n"

    return s, explanation


def grad_sigmoid(x, help=False, what=False):

    s = 1 / (1 + np.exp(-x))
    grad = s * (1 - s)
    explanation = ""

    if what:
        explanation += "WHAT: Gradient of sigmoid tells slope of curve.\n"
        explanation += "Formula: sigmoid(x) * (1 - sigmoid(x))\n"
        explanation += f"Example: grad_sigmoid({x}) = {grad}\n"
        return grad, explanation

    if help:
        explanation += "Sigmoid gradient:\n"
        explanation += "Step 1: Compute sigmoid s = 1 / (1 + exp(-x))\n"
        explanation += "Step 2: Gradient = s * (1 - s)\n"
        explanation += f"Input: {x}\nSigmoid: {s}\nGradient: {grad}\n"

    return grad, explanation


def softmax(x, axis=-1, help=False, what=False):

    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    result = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    explanation = ""

    if what:
        explanation += "WHAT: Softmax converts numbers into probabilities.\n"
        explanation += "Each value becomes exp(x) / sum(exp(x)).\n"
        explanation += f"Example: softmax({x.tolist()}) = {result.tolist()}\n"
        return result, explanation

    if help:
        explanation += "Softmax activation:\n"
        explanation += "Step 1: Shift inputs by max for numerical stability\n"
        explanation += f"Shifted input:\n{x_shifted}\n"
        explanation += "Step 2: Apply exp\n"
        explanation += f"exp values:\n{exp_x}\n"
        explanation += "Step 3: Divide by sum of exp values\n"
        explanation += f"Output:\n{result}\n"

    return result, explanation


def cross_entropy(pred, target, help=False, what=False):

    eps = 1e-12
    pred_clipped = np.clip(pred, eps, 1 - eps)
    loss = -np.sum(target * np.log(pred_clipped)) / pred.shape[0]
    explanation = ""

    if what:
        explanation += "WHAT: Cross-entropy measures how different prediction is from target.\n"
        explanation += "Used for classification.\n"
        explanation += f"Example: cross_entropy({pred.tolist()}, {target.tolist()}) = {loss}\n"
        return loss, explanation

    if help:
        explanation += "Cross-entropy loss step-by-step:\n"
        explanation += "Step 1: Clip predictions to avoid log(0)\n"
        explanation += f"Clipped predictions:\n{pred_clipped}\n"
        explanation += "Step 2: Compute -sum(target * log(pred)) / batch size\n"
        explanation += f"Target:\n{target}\nLoss: {loss}\n"

    return loss, explanation
