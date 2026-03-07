import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

export default function DecompsPage() {
    return (
        <>
            <Head><title>det, inv, qr, eig — ML-Lens math.linalg</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'math.linalg', href: '/guide/linalg' }, { label: 'Decompositions', href: '/guide/linalg/det' }]}
                toc={[{ id: 'det', label: 'det' }, { id: 'inv', label: 'inv' }, { id: 'trace', label: 'matrix_trace' }, { id: 'qr', label: 'qr' }, { id: 'eig', label: 'eig' }, { id: 'dominant', label: 'dominant_eigen' }]}
                prev={{ label: 'Shape Ops', href: '/guide/linalg/transpose' }}
                next={{ label: 'svd', href: '/guide/linalg/svd' }}
            >
                <h1>Matrix Decompositions</h1>
                <p>Factorization and decomposition operations for square and rectangular matrices.</p>

                <h2 id="det">det — Determinant</h2>
                <p>The determinant of a square matrix is a scalar that encodes whether the matrix is invertible (det ≠ 0) and describes the volume scaling factor of the linear transformation.</p>
                <CodeBlock code={`from mllense.math.linalg import det
A = [[1, 2], [3, 4]]
d = det(A, what_lense=True)
print(d)  # → -2.0
# det = 1*4 - 2*3 = -2`} />

                <h2 id="inv">inv — Matrix Inverse</h2>
                <p>Computes A⁻¹ such that A @ A⁻¹ = I (identity). Only exists for square, non-singular matrices (det ≠ 0).</p>
                <CodeBlock code={`from mllense.math.linalg import inv
A = [[2, 1], [5, 3]]
A_inv = inv(A, how_lense=True)
print(A_inv)
# → [[3, -1], [-5, 2]]`} />

                <h2 id="trace">matrix_trace</h2>
                <p>Sum of the diagonal elements of a square matrix. Equals the sum of all eigenvalues.</p>
                <CodeBlock code={`from mllense.math.linalg import matrix_trace
A = [[1, 2], [3, 4]]
tr = matrix_trace(A)
print(tr)  # → 5.0 (1 + 4)`} />

                <h2 id="qr">qr — QR Decomposition</h2>
                <p>Factors A = QR where Q is orthogonal (Q^T Q = I) and R is upper triangular. Used in least squares, eigenvalue algorithms.</p>
                <CodeBlock code={`from mllense.math.linalg import qr
A = [[1, 2], [3, 4], [5, 6]]  # (3x2)
Q, R = qr(A, what_lense=True, how_lense=True)
print(Q.shape)  # (3, 3) — orthogonal
print(R.shape)  # (3, 2) — upper triangular`} />

                <h2 id="eig">eig — Eigendecomposition</h2>
                <p>Computes eigenvalues λ and eigenvectors v such that Av = λv. A must be square.</p>
                <CodeBlock code={`from mllense.math.linalg import eig
A = [[4, 1], [2, 3]]
eigenvalues, eigenvectors = eig(A, what_lense=True)
print(eigenvalues)   # [5.0, 2.0]
print(eigenvectors)  # Corresponding eigenvectors`} />

                <h2 id="dominant">dominant_eigen</h2>
                <p>Power iteration method to find the largest eigenvalue and its eigenvector. Faster than full eigendecomposition when only the dominant eigenpair is needed.</p>
                <CodeBlock code={`from mllense.math.linalg import dominant_eigen
A = [[4, 1], [2, 3]]
eigenvalue, eigenvector = dominant_eigen(A, max_iter=100, how_lense=True)
print(eigenvalue)   # ≈ 5.0 (largest eigenvalue)
print(eigenvector)  # Corresponding eigenvector`} />
            </GuideLayout>
        </>
    );
}
