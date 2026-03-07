import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

const toc = [
    { id: 'overview', label: 'Overview' },
    { id: 'matmul', label: 'matmul' },
    { id: 'solve', label: 'solve' },
    { id: 'ops', label: 'Element-wise Ops' },
    { id: 'shape', label: 'Shape Operations' },
    { id: 'decomp', label: 'Decompositions' },
    { id: 'norms', label: 'Norms' },
    { id: 'diag', label: 'Diagnostics' },
    { id: 'creation', label: 'Creation' },
];

export default function LinalgGuide() {
    return (
        <>
            <Head><title>math.linalg — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'math.linalg', href: '/guide/linalg' }]}
                toc={toc}
                prev={{ label: 'KMeans', href: '/guide/models/kmeans' }}
                next={{ label: 'matmul', href: '/guide/linalg/matmul' }}
            >
                <h1 id="overview">math.linalg</h1>
                <blockquote>A production-grade, educational, extensible linear algebra engine — NumPy API compatible with full tracing support.</blockquote>

                <p>The <code className="inline-code">mllense.math.linalg</code> module provides a complete linear algebra toolkit with 27+ operations. It is designed to be:</p>
                <ul>
                    <li><strong>NumPy-compatible</strong> — same function names and signatures</li>
                    <li><strong>Multi-backend</strong> — switch between numpy, python, or numba backends</li>
                    <li><strong>Fully traceable</strong> — every operation returns a <code className="inline-code">LinalgResult</code></li>
                    <li><strong>Auto-selecting</strong> — algorithm registry picks optimal implementation based on matrix size</li>
                </ul>

                <h2 id="matmul">matmul — Matrix Multiplication</h2>
                <p>The backbone of neural networks, linear regression, and all matrix operations. Supports 1D×1D, 2D×1D, and 2D×2D inputs.</p>
                <CodeBlock code={`from mllense.math.linalg import matmul\nresult = matmul(A, B, what_lense=True, how_lense=True)\nprint(result.how_lense)\n# → "1. Validated shapes: A(2x2) @ B(2x2) -> Result(2x2).\n#    2. Triple-loop computation: Result[0][0] += A[0][0] * B[0][0]..."`} />

                <h2 id="solve">solve — Linear System Solver</h2>
                <p>Solves Ax=b using Gaussian elimination with partial pivoting.</p>
                <CodeBlock code={`from mllense.math.linalg import solve\nx = solve([[3, 1], [1, 2]], [9, 8])\nprint(x)  # → [2.0, 3.0]`} />

                <h2 id="ops">Element-wise Operations</h2>
                <CodeBlock code={`from mllense.math.linalg import add, subtract, multiply, divide, scalar_add, scalar_multiply\nC = add(A, B)\nC = subtract(A, B)\nC = multiply(A, B)       # Hadamard (element-wise) product\nC = divide(A, B)\nC = scalar_multiply(A, 3.0)\nC = scalar_add(A, 5.0)`} />

                <h2 id="shape">Shape Operations</h2>
                <CodeBlock code={`from mllense.math.linalg import transpose, reshape, flatten, vstack, hstack\nAt = transpose(A)\nB = reshape(A, (4, 2))\nv = flatten(A)\nM = vstack((A, B))\nM = hstack((A, B))`} />

                <h2 id="decomp">Matrix Decompositions</h2>
                <CodeBlock code={`from mllense.math.linalg import det, inv, matrix_trace, qr, svd, eig, dominant_eigen\nd = det(A)\nA_inv = inv(A)\ntr = matrix_trace(A)\nQ, R = qr(A)\nU, S, Vt = svd(A)\neigenvalues, eigenvectors = eig(A)\nval, vec = dominant_eigen(A, max_iter=100)`} />

                <h2 id="norms">Norms</h2>
                <CodeBlock code={`from mllense.math.linalg import vector_norm, frobenius_norm, spectral_norm\nn = vector_norm(v, ord=2)   # L2 norm of vector\nf = frobenius_norm(A)       # sqrt of sum of all squared elements\ns = spectral_norm(A)        # largest singular value`} />

                <h2 id="diag">Diagnostics</h2>
                <CodeBlock code={`from mllense.math.linalg import condition_number, matrix_rank, stability_report, full_diagnostic_report\ncn = condition_number(A)           # σ_max / σ_min via SVD\nrk = matrix_rank(A)                # rank via SVD threshold\nsr = stability_report(A)           # perturbation analysis\nfdr = full_diagnostic_report(A)    # all diagnostics combined`} />

                <h2 id="creation">Matrix Creation</h2>
                <CodeBlock code={`from mllense.math.linalg import zeros, ones, eye, rand\nZ = zeros((3, 4))\nO = ones((2, 2))\nI = eye(4)\nR = rand((5, 5))`} />
            </GuideLayout>
        </>
    );
}
