# ==============================
# Tutorial 9: Neural Network Primitives
# ==============================
"""
This tutorial uses the native mathematical primitives within the internal `nn` 
sub-package to demonstrate how fundamental neural net blocks function.
These operate completely native to the linalg framework.
"""

from mllense.math.linalg.nn import (
    relu, sigmoid, tanh, softmax,
    linear_forward,
    mse_loss, cross_entropy_loss,
    numerical_gradient
)

def run_tutorial():
    print("=== Linalg Tutorial: Neural Network Primitives ===")

    # ── 1. Activations ──
    v = [-2.0, 0.0, 1.0, 2.0]
    print(f"Input Vector: {v}\n")
    
    # ReLU clips negative values to 0
    print(f"ReLU:    {[round(x, 2) for x in relu(v)]}")
    
    # Sigmoid squeezes values between 0 and 1
    print(f"Sigmoid: {[round(x, 2) for x in sigmoid(v)]}")
    
    # Tanh squeezes values between -1 and 1
    print(f"Tanh:    {[round(x, 2) for x in tanh(v)]}")
    
    # Softmax scales values to a probability distribution (sum = 1)
    smax = softmax(v)
    print(f"Softmax: {[round(x, 2) for x in smax]} (sum: {sum(smax):.2f})")

    # ── 2. Linear Layer (Forward Pass) ──
    # Input batch X: 2 samples, 3 features
    X = [[1.0, -1.0, 2.0],
         [0.0,  2.0, 1.0]]
         
    # Weights W: 2 out features, 3 in features
    W = [[0.5, 0.5, 0.5],
         [-0.5, 1.0, 0.0]]
         
    # Bias (length 2)
    b = [0.1, -0.1]
    
    # Computes X @ W^T + b
    Y = linear_forward(X, W, bias=b)
    
    print("\nLinear Forward Output (X @ W^T + b):")
    for row in Y: print("  ", [round(x, 2) for x in row])

    # ── 3. Losses ──
    pred = [0.9, 0.1, 0.8]
    targets = [1.0, 0.0, 1.0]
    
    mse = mse_loss(pred, targets)
    ce = cross_entropy_loss(pred, targets)
    print(f"\nPredictions: {pred}")
    print(f"Targets:     {targets}")
    print(f"MSE Loss:           {mse:.4f}")
    print(f"Cross Entropy Loss: {ce:.4f}")

    # ── 4. Numerical Gradients ──
    # We can compute the functional gradient of any python function using limits!
    # Let f(x) = x_0^2 + 3*x_1
    def f(x): return x[0]**2 + 3*x[1]
    
    val = [2.0, 1.0] 
    # Analytical grad is [2*x_0, 3] -> At x=[2, 1], grad should be [4, 3]
    
    grad = numerical_gradient(f, val)
    print(f"\nNumerical Gradient of f(x) at {val}:")
    print(f"  {grad}")

if __name__ == "__main__":
    run_tutorial()
