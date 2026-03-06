# ==============================
# File: linalg/utils/block_ops.py
# ==============================
"""Block matrix operations for cache-friendly algorithms."""

from __future__ import annotations

from mllense.math.linalg.core.types import InternalMatrix
from mllense.math.linalg.exceptions import InvalidInputError

__all__ = ["split_into_blocks", "merge_blocks"]


def split_into_blocks(
    m: InternalMatrix, block_rows: int, block_cols: int
) -> list[list[InternalMatrix]]:
    """Split a matrix into a grid of sub-blocks.

    Args:
        m: Input matrix (``rows × cols``).
        block_rows: Number of rows per block.
        block_cols: Number of columns per block.

    Returns:
        A 2-D list where ``result[bi][bj]`` is the sub-block matrix.
        The last row/column of blocks may be smaller if dimensions are
        not evenly divisible.
    """
    if block_rows <= 0 or block_cols <= 0:
        raise InvalidInputError("Block size must be positive.")

    rows = len(m)
    cols = len(m[0]) if rows else 0

    grid: list[list[InternalMatrix]] = []
    for i in range(0, rows, block_rows):
        row_blocks: list[InternalMatrix] = []
        for j in range(0, cols, block_cols):
            block: InternalMatrix = []
            for r in range(i, min(i + block_rows, rows)):
                block.append(m[r][j: min(j + block_cols, cols)])
            row_blocks.append(block)
        grid.append(row_blocks)
    return grid


def merge_blocks(grid: list[list[InternalMatrix]]) -> InternalMatrix:
    """Merge a grid of sub-blocks back into a single matrix.

    Inverse of :func:`split_into_blocks`.
    """
    result: InternalMatrix = []
    for block_row in grid:
        if not block_row:
            continue
        num_rows_in_block = len(block_row[0])
        for r in range(num_rows_in_block):
            row: list[float] = []
            for block in block_row:
                if r < len(block):
                    row.extend(block[r])
            result.append(row)
    return result
