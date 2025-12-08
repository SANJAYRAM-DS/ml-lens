import random

def zeros(shape):
    rows, cols = shape
    return [[0 for _ in range(cols)] for _ in range(rows)]

def ones(shape):
    rows, cols = shape
    return [[1 for _ in range(cols)] for _ in range(rows)]

def eye(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def rand(shape, seed=None):
    if seed is not None:
        random.seed(seed)
    rows, cols = shape
    return [[random.random() for _ in range(cols)] for _ in range(rows)]


print(rand((2, 2), seed=42))