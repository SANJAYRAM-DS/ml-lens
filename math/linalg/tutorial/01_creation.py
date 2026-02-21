# ==============================
# Tutorial 1: Matrix Creation
# ==============================
"""
This tutorial covers the `create` API in the mllense.math.linalg package.
You can use these functions to instantiate native Python lists or NumPy arrays
representing mathematical matrices and vectors.
"""

from mllense.math.linalg import zeros, ones, eye, rand

def run_tutorial():
    print("=== Linalg Tutorial: Matrix Creation ===")

    # 1. Zeros
    # Creates a matrix filled with 0.0
    z = zeros(2, 3)
    print("\n1. zeros(2, 3):")
    for row in z:
        print("  ", row)

    # 2. Ones
    # Creates an array filled with 1.0. If you provide just one dimension, it creates a vector.
    o = ones(4)
    print("\n2. ones(4):")
    print("  ", o)

    # 3. Identity Matrix (Eye)
    # Creates a 2D matrix with 1.0 on the diagonal and 0.0 elsewhere.
    e = eye(3)
    print("\n3. eye(3):")
    for row in e:
        print("  ", row)

    # 4. Random Matrix
    # Creates a matrix with values uniformly distributed between 0.0 and 1.0.
    # A seed can be provided for reproducibility.
    r = rand(2, 2, seed=42)
    print("\n4. rand(2, 2, seed=42):")
    for row in r:
        print("  ", row)

    # 5. Output as NumPy array
    # Any of these functions can return a NumPy array directly if requested.
    z_np = zeros(2, 2, as_numpy=True)
    print("\n5. zeros(2, 2, as_numpy=True):")
    print("  ", type(z_np))
    print(z_np)

if __name__ == "__main__":
    run_tutorial()