"""Create an identity matrix."""

from __future__ import annotations

from typing import Any

from mllense.math.linalg.algorithms.creation.base import BaseCreation
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix
from mllense.math.linalg.exceptions import InvalidInputError

__all__ = ["EyeCreation"]


class EyeCreation(BaseCreation):
    """Create an ``n × m`` identity-like matrix (ones on the main diagonal)."""

    metadata = AlgorithmMetadata(
        name="eye",
        operation="create_eye",
        complexity="O(n*m)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Creates an identity (or rectangular identity-like) matrix.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        """Create identity matrix.

        Args:
            args[0]: rows (int)
            args[1]: cols (int, defaults to rows)
        """
        rows: int = args[0]
        cols: int = args[1] if len(args) > 1 else rows

        if rows <= 0 or cols <= 0:
            raise InvalidInputError(
                f"Matrix dimensions must be positive, got ({rows}, {cols})."
            )

        trace.record(
            operation="eye",
            description=f"Creating {rows}×{cols} identity matrix",
        )

        what = context.what_lense_enabled
        how = context.how_lense_enabled

        if what:
            self.what_lense = (
                "=== WHAT: Identity Matrix ===\n"
                "An Identity Matrix (often denoted as I) is a square matrix that has 1s on its "
                "main diagonal and 0s everywhere else. It acts as the 'number 1' in matrix math, "
                "meaning if you multiply any matrix A by I, you get back A.\n\n"
                "=== WHY we need it in ML ===\n"
                "It serves as a fundamental building block for matrix algebra operations, giving us a neutral "
                "starting point. When defining inversions or solving equations, we rely on the identity matrix "
                "as a target (like A * A^-1 = I).\n\n"
                "=== WHERE it is used in Real ML (Real-time examples) ===\n"
                "1. Regularization (Ridge Regression): It is used in the L2 penalty term to ensure the matrix is "
                "invertible (X^T * X + lambda * I). It is also used heavily in covariance stabilization in Kalman filters and Gaussian models.\n"
                "2. Weight Initialization: Sometimes linear layers in neural networks are initialized close to the "
                "identity matrix (Identity Initialization) so the network starts off acting like a transparent pass-through.\n"
                "3. Algorithms: In Kalman filters and attention mechanisms, it helps compute scaled properties without "
                "changing coordinate values prematurely."
            )
        else:
            self.what_lense = ""

        if how:
            checkpoints = []
            checkpoints.append(f"1. Validated inputs: {rows} rows, {cols} columns.")
            checkpoints.append(f"2. Checked if dimensions are valid. (They are).")
            checkpoints.append(f"3. Beginning loop over {rows} rows to populate matrix:")

        total = rows * cols
        step_count = 0
        omitted = False

        matrix = []
        for i in range(rows):
            row_data = []
            for j in range(cols):
                step_count += 1
                val = 1.0 if i == j else 0.0
                row_data.append(val)
                
                if how:
                    if total <= 10 or step_count <= 5 or step_count > total - 5:
                        checkpoints.append(f"   - Row {i}, Col {j}: i == j is {i == j} -> placed {val}")
                    elif not omitted:
                        checkpoints.append("   - ... (skipped intermediate values) ...")
                        omitted = True

            matrix.append(row_data)
            
        if how:
            checkpoints.append("4. Finished building the matrix and returning it.")
            self.how_lense = "\n".join(checkpoints)
        else:
            self.how_lense = ""

        return matrix