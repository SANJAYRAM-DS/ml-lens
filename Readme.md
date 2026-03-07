<div align="center">

# mllense

### Understand Machine Learning — Not Just Use It.

A production-grade, educational ML and linear algebra framework with built-in **explainability lenses** that reveal *how every computation really works*.

[![PyPI version](https://img.shields.io/pypi/v/mllense.svg)](https://pypi.org/project/mllense/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://pypi.org/project/mllense/)
[![Status](https://img.shields.io/badge/status-active-success.svg)](https://pypi.org/project/mllense/)

</div>

---

## ✨ What is mllense?

**mllense** is an open-source Python library that makes machine learning **transparent, educational, and deeply understandable**.

Every function and model ships with two built-in lenses:

- **`what_lense`** — Explains *what* the algorithm is, *why* it exists, and *where* it's used in real ML pipelines.
- **`how_lense`** — Traces *exactly* what happened computationally: shape checks, arithmetic steps, loop iterations, convergence.

If scikit-learn helps you **build models**, mllense helps you **understand them**.

---

## 🚀 Installation

```bash
pip install mllense
```

---

## ⚡ Quick Start

### Linear Algebra (`mllense.math.linalg`)

```python
from mllense.math.linalg import matmul, det, eig, svd, solve

# Matrix multiplication
C = matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
print(C)  # [[19.0, 22.0], [43.0, 50.0]]

# With explainability lenses
result = matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]], what_lense=True, how_lense=True)
print(result.what_lense)  # "=== WHAT: Matrix Multiplication ==="
print(result.how_lense)   # Step-by-step computation trace
```

### ML Models (`mllense.models`)

```python
from mllense.models import LinearRegression, LogisticRegression, KMeans

model = LinearRegression(what_lense=True, how_lense=True)
model.fit(X, y)
result = model.predict(X)

print(result.what_lense)  # Mathematical explanation
print(result.how_lense)   # Training step trace
```

---

## 🔍 The Lenses

### `what_lense` — Theoretical Context

Every result carries a human-readable explanation of:
- 🧠 **WHAT** the algorithm is mathematically
- 💡 **WHY** it exists and what problem it solves
- 🏭 **WHERE** it's used in real ML pipelines

```python
result = matmul(A, B, what_lense=True)
print(result.what_lense)
# === WHAT: Matrix Multiplication ===
# Matrix multiplication combines the rows of the first matrix with
# the columns of the second. Each element is the sum of products
# of corresponding elements.
#
# === WHY we need it in ML ===
# It allows computing many linear combinations at once — the core
# of feed-forward neural networks (Weights * Inputs), attention
# mechanisms (Q * K^T), and embedding projections.
#
# === WHERE it is used in Real ML ===
# 1. Dense/Linear Layers: Output = Weights @ Inputs + Bias.
# 2. Convolutions: Often lowered to matmul (im2col).
# 3. Transformers: Self-attention relies on batched matmuls.
```

### `how_lense` — Operational Trace

Shows exactly what happened step-by-step:

```python
result = matmul(A, B, how_lense=True)
print(result.how_lense)
# 1. Validated shapes: A(2x2) @ B(2x2) -> Result(2x2).
# 2. Initialized empty output matrix of shape 2x2.
# 3. Triple-loop computation (m=2, k=2, n=2):
#    - Result[0][0] += A[0][0] * B[0][0] (1 * 5 = 5)
#    - Result[0][0] += A[0][1] * B[1][0] (2 * 7 = 14)
#    ...
# 4. Finished matrix multiplication.
```

---

## 📦 Full API Reference

### `mllense.math.linalg`

| Category | Functions |
|---|---|
| **Matrix Ops** | `matmul`, `add`, `subtract`, `multiply`, `divide`, `scalar_add`, `scalar_multiply` |
| **Creation** | `zeros`, `ones`, `eye`, `rand` |
| **Shape** | `transpose`, `reshape`, `flatten`, `vstack`, `hstack` |
| **Decomposition** | `det`, `inv`, `qr`, `svd`, `eig`, `matrix_trace` |
| **Solver** | `solve` |
| **Eigen** | `dominant_eigen` |
| **Norms** | `vector_norm`, `frobenius_norm`, `spectral_norm` |
| **Diagnostics** | `condition_number`, `matrix_rank`, `stability_report`, `full_diagnostic_report` |
| **Config** | `get_config`, `GlobalConfig` |
| **Constants** | `constants` |

### `mllense.models`

| Model | Import |
|---|---|
| Linear Regression | `from mllense.models import LinearRegression` |
| Logistic Regression | `from mllense.models import LogisticRegression` |
| Decision Tree Classifier | `from mllense.models import DecisionTreeClassifier` |
| Decision Tree Regressor | `from mllense.models import DecisionTreeRegressor` |
| Random Forest Classifier | `from mllense.models import RandomForestClassifier` |
| Random Forest Regressor | `from mllense.models import RandomForestRegressor` |
| K-Means Clustering | `from mllense.models import KMeans` |

---

## 🎛️ Global Configuration

```python
from mllense.math.linalg import GlobalConfig

GlobalConfig.default_backend = "numpy"   # "numpy" | "python"
GlobalConfig.default_mode = "fast"        # "fast" | "educational"
GlobalConfig.trace_enabled = True         # Enable how_lense globally
```

---

## 📊 The Result Object

All operations return a result object that behaves **exactly like the raw value** but carries trace metadata:

```python
result = matmul(A, B, what_lense=True, how_lense=True)

# Acts like a plain matrix
print(result)            # [[19.0, 22.0], [43.0, 50.0]]
print(result[0])         # [19.0, 22.0]
print(len(result))       # 2

# Educational metadata attached
print(result.what_lense)     # str — theoretical explanation
print(result.how_lense)      # str — computation trace
print(result.algorithm_used) # "naive_matmul"
print(result.complexity)     # "O(m*k*n)"
```

---

## 🧱 Architecture

```
mllense/
├── math/
│   └── linalg/          # Linear algebra engine
│       ├── api/          # Clean public API
│       ├── algorithms/   # Core implementations (matmul, solve, decomp...)
│       ├── backend/      # NumPy / Python backends
│       ├── core/         # Execution context, tracing
│       ├── diagnostics/  # Condition number, rank, stability
│       ├── nn/           # Neural network primitives
│       └── registry/     # Algorithm & backend registry
└── models/              # ML models
    ├── linear_model/     # LinearRegression, LogisticRegression
    ├── tree/             # DecisionTree
    ├── ensemble/         # RandomForest
    └── cluster/          # KMeans
```

---

## 🆚 How is this different?

| Tool | Focus |
|---|---|
| scikit-learn | Model building |
| SHAP / LIME | Feature attribution |
| TensorBoard | Training metrics |
| **mllense** | **First-principles understanding of every step** |

---

## 🤝 Contributing

We welcome contributors from ML research, education, and open-source.

```bash
git clone https://github.com/SANJAYRAM-DS/ml-lens
cd ml-lens
pip install -e ".[dev]"
```

Ways to contribute:
- Add new algorithm implementations
- Improve what_lense explanations
- Expand model support
- Write tutorials

---

## 📄 License

MIT License — use freely in research, education, and industry.

---

<div align="center">

Built with clarity in mind. | [PyPI](https://pypi.org/project/mllense/) · [GitHub](https://github.com/SANJAYRAM-DS/ml-lens)

</div>