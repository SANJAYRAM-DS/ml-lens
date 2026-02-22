# ==============================
# Tutorial 10: Tracing and Execution Context
# ==============================
"""
This tutorial demonstrates how to leverage tracing and algorithm overriding securely.
The linalg engine supports deep introspection, meaning we can natively "trace" 
and summarize the computational steps applied dynamically, including loops and complexities.
"""

from mllense.math.linalg import matmul
from mllense.math.linalg.trace import Trace

def run_tutorial():
    print("=== Linalg Tutorial: Execution Tracing ===")

    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]

    # 1. Tracing in action!
    print("We will multiply two 2x2 matrices with trace_enabled=True.")
    
    # Using the exact API parameter `trace_enabled=True` causes internal 
    # `Algorithm` classes to push records to a central Trace object.
    # Note: normally we would retrieve the trace directly from the engine, 
    # but for typical API usages, the Trace object is injected by the developer via core,
    # or you can simply set the global config.
    from mllense.math.linalg.config import get_config
    cfg = get_config()
    
    print("\nGlobal Config Values:")
    print(f"  Backend: {cfg.default_backend}")
    print(f"  Mode: {cfg.default_mode}")
    print(f"  Tracing Global: {cfg.trace_enabled}")
    
    # We're manually showing the result mapping here for typical operations.
    res = matmul(A, B)
    print(f"Result: {res}")
    
    # To use a raw Trace directly, you interact with the backend or algorithms themselves:
    from mllense.math.linalg.algorithms.matmul.naive import NaiveMatmul
    from mllense.math.linalg.core.execution_context import ExecutionContext
    from mllense.math.linalg.core.mode import ExecutionMode
    
    ctx = ExecutionContext("python", ExecutionMode.EDUCATIONAL, trace_enabled=True)
    t = Trace(enabled=True)
    
    NaiveMatmul().execute(A, B, context=ctx, trace=t)
    
    print("\n--- Detailed Educational Trace output ---")
    print(t.replay())


    # --- Educational Expose ---
    print("\n=== EXPLAINABILITY SHOWCASE ===")
    from mllense.math.linalg import eye
    demo = eye(2, how_lense=True)
    print("- [WHAT LENSE]:", demo.what_lense.split('\n')[0])
    print("- [HOW LENSE]:", demo.how_lense.split('\n')[0], "...")

if __name__ == "__main__":
    print('\nRunning with Educational Engine v2.0\n')
run_tutorial()
