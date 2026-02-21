PRODUCT REQUIREMENTS DOCUMENT
Project: linalg-project
Goal: Production-grade, educational, extensible linear algebra engine
1️⃣ Product Vision

Build a:

🔹 NumPy-like API

🔹 Multi-backend execution engine

🔹 Algorithm-switching system

🔹 Educational traceable computation engine

🔹 ML-ready math core

This is NOT a toy.
This is a computational math framework.

2️⃣ Core Architectural Principles
A. Layered Architecture (Strict)
API
 ↓
Core
 ↓
Algorithms
 ↓
Backend

Rules:

API must NOT import algorithms directly

Algorithms must NOT import API

Backend must NOT import API

Core can be imported by everyone

Registry connects everything

3️⃣ Root-Level Files
📄 pyproject.toml

Use:

build backend: hatchling or setuptools

dependencies:

numpy

numba (optional)

scipy (optional)

pytest

pydantic (for validation if needed)

Add:

optional extras: ["numba", "scipy"]

📄 README.md

Must include:

Architecture diagram

Backend system explanation

Educational mode explanation

Algorithm switching explanation

Performance benchmarks

📄 .pre-commit-config.yaml

Include:

black

isort

flake8

mypy

4️⃣ docs/

Pure documentation.

Must include:

diagrams (Mermaid)

architecture flow

algorithm comparison tables

complexity breakdown tables

5️⃣ benchmarks/

Use:

time.perf_counter

numpy as reference baseline

Each benchmark must compare:

python backend

numpy backend

numba backend

Include matrix sizes:

10x10

100x100

1000x1000

6️⃣ examples/

Must demonstrate:

switching backends

enabling educational mode

comparing algorithms

mini NN training

7️⃣ LINALG CORE ENGINE

Now we go file-by-file.

🔹 linalg/init.py

Must:

expose public API

initialize backend registry

initialize algorithm registry

call auto_register()

🔹 config.py

Must contain:

class GlobalConfig:
    default_backend = "numpy"
    default_mode = "fast"
    trace_enabled = False
    auto_algorithm_selection = True

Edge cases:

thread safety

dynamic config update

🔹 exceptions.py

Define:

ShapeMismatchError

SingularMatrixError

InvalidBackendError

AlgorithmNotFoundError

InvalidModeError

Never use raw ValueError.

🔹 version.py

Auto-populated from package metadata.

8️⃣ API Layer

Thin wrappers only.

Each API file:

validate inputs

create execution context

call registry to fetch algorithm

execute algorithm

return result

NO heavy logic here.

Example: api/matmul.py

Must behave like numpy.matmul:

Handle:

1D x 1D → scalar

2D x 2D → matrix

2D x 1D → vector

broadcasting

batch matmul

Edge cases:

non-numeric input

zero-size matrices

mismatched shapes

large dimension overflow

Must delegate to:

algorithm_registry.get("matmul", context)
9️⃣ CORE/

Infrastructure layer.

types.py

Define:

MatrixLike = Union[List[List[float]], np.ndarray]
VectorLike = Union[List[float], np.ndarray]

Must enforce:

numeric types only

rectangular matrices

validation.py

Must validate:

shape compatibility

square matrix requirement

positive definiteness (for Cholesky)

rank checks

Never trust user input.

execution_context.py

Must hold:

backend

mode (educational/fast)

algorithm hint

trace flag

Immutable object.

trace.py

Must collect:

step-by-step operations

intermediate matrices

complexity notes

Educational mode must:

store operation logs

allow replay

complexity.py

Must compute:

theoretical complexity

actual operation count

memory usage estimate

metadata.py

Each algorithm must declare:

class AlgorithmMetadata:
    name
    complexity
    stable
    supports_batch
    requires_square
mode.py

Define:

FAST

EDUCATIONAL

DEBUG

🔟 ALGORITHMS/

This is heart of system.

Each algorithm file must:

inherit from BaseAlgorithm

implement execute()

declare metadata

Example: matmul/naive.py

Must implement triple-loop multiplication.

Edge cases:

empty matrices

1D inputs

overflow

float precision accumulation

Bottlenecks:

O(n^3)

poor cache usage

matmul/block.py

Must:

split matrices

optimize cache locality

handle uneven block sizes

Edge cases:

dimensions not divisible by block size

matmul/strassen.py

Must:

handle power-of-two padding

fallback to naive for small sizes

avoid excessive recursion

Major bottleneck:

memory overhead

numerical instability

solve/gaussian.py

Must:

implement partial pivoting

detect singular matrices

avoid division by zero

Edge cases:

near-zero pivots

float instability

decomposition/qr.py

Implement:

Gram-Schmidt (classical + modified)

Householder

Modified Gram-Schmidt preferred.

decomposition/svd.py

Wrap numpy OR implement power iteration based SVD.

Note:

Full SVD from scratch is very complex.

eigen/power_iteration.py

Must:

normalize vector

detect convergence

prevent infinite loop

Edge cases:

non-dominant eigenvalue scenario

degenerate eigenvalues

BACKEND/

Abstraction layer.

Each backend must implement:

class Backend:
    matmul()
    dot()
    solve()
    inverse()
numpy_backend.py

Use:

numpy.matmul

numpy.linalg.solve

numpy.linalg.inv

But:

Must convert input/output to library format.

python_backend.py

Pure Python list operations.

numba_backend.py

Use:

@njit

Edge cases:

object mode fallback

compilation warmup delay

NN/

Built on top of linalg.

Must use your API layer.

Never directly use numpy.

DIAGNOSTICS/

Condition number:

use SVD

Rank:

based on SVD threshold

Stability:

perturbation analysis

REGISTRY/

Central plug-in system.

algorithm_registry.py

Store:

{
  "matmul": {
      "naive": Class,
      "block": Class,
      ...
  }
}

Auto-selection logic:

small matrix → naive

medium → block

large → numpy backend

backend_registry.py

Register:

python

numpy

numba

auto_register.py

Scan:

algorithms package

use importlib

auto-register subclasses

UTILS/

Performance:

benchmark wrappers

memory profiling

Logging:

structured logging

Inspection:

introspection tools

_internal/

Never exposed.

Decorators:

@algorithm

@register_backend

Constants:

tolerance values

TESTS/

Each test must include:

correctness

edge cases

precision tolerance

stress test

random test

cross-check with numpy

Example:

assert np.allclose(my_matmul(A, B), np.matmul(A, B))
⚠️ Bottlenecks

Memory allocation

Python loop overhead

Recursion depth (Strassen)

Floating-point accumulation error

Backend conversion overhead

⚠️ Critical Edge Cases

Empty matrices

1xN and Nx1 shapes

Singular matrices

Near-singular matrices

Very large matrices

Float32 vs Float64 mismatch

Non-contiguous arrays

Integer overflow

Zero division

Negative strides

Broadcasting mismatch

🔥 How to Model After NumPy

Study:

numpy/core/src/multiarray/matmul.c

numpy/linalg/linalg.py

Key ideas:

vectorized operations

memory contiguity

fallback algorithms

BLAS usage

But you:

Use numpy for fast backend

Implement educational algorithms in Python

🚀 Performance Strategy

Auto-select algorithm

Use block multiplication

JIT with numba

Avoid copying arrays

Reuse buffers

Preallocate output matrix

🏆 Final Architecture Summary

This engine is:

API Layer

Execution context

Algorithm registry

Backend registry

Trace system

Diagnostics

NN built on top


It is a modular computational math framework.