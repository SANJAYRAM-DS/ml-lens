import numpy as np
from ..gradients import grad

def gradient_descent(f, x0, lr=0.01, epochs=100, help=False, what=False):
    """
    Basic gradient descent optimizer.
    """
    x = np.array(x0, dtype=float)

    steps = []
    if help:
        steps.append("Gradient Descent:\n x_new = x - lr * grad(f,x)")

    for e in range(epochs):
        g = grad(f, x)
        x -= lr * g
        if help:
            steps.append(f"Epoch {e+1}: grad={g}, x={x}")

    what_text = ("""
Gradient Descent iteratively moves opposite to gradient to minimize a function.

Example:
    f(x) = x²
    Starting from x=3 → GD converges toward x=0.
""") if what else None

    if help or what:
        return x, ("\n".join(steps) if help else None), what_text
    return x


def momentum_optimizer(grad_f, x0, lr=0.01, beta=0.9, epochs=100, help=False, what=False):
    """
    Gradient Descent with Momentum.
    """
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)

    steps = []
    if help:
        steps.append("Momentum: v = βv + lr*g ; x = x - v")

    for e in range(epochs):
        g = grad_f(x)
        v = beta * v + lr * g
        x -= v
        if help:
            steps.append(f"Epoch {e+1}: grad={g}, velocity={v}, x={x}")

    what_text = ("""
Momentum smooths updates by adding velocity.
Helps escape shallow minima and reduces oscillation.
""") if what else None

    if help or what:
        return x, ("\n".join(steps) if help else None), what_text
    return x


def adam_optimizer(grad_f, x0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                   epochs=100, help=False, what=False):
    """
    Adam optimizer (adaptive moment estimation)
    """
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    steps = []
    if help:
        steps.append("Adam: combines Momentum + RMSProp")

    for t in range(1, epochs+1):
        g = grad_f(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if help:
            steps.append(f"Epoch {t}: g={g}, m={m_hat}, v={v_hat}, x={x}")

    what_text = ("""
Adam adapts per-parameter learning rates using first and second moment estimates.
Commonly used default optimizer for deep learning.
""") if what else None

    if help or what:
        return x, ("\n".join(steps) if help else None), what_text
    return x
