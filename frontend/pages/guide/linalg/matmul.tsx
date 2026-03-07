import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

const toc = [{ id: 'what', label: 'What is matmul?' }, { id: 'ml', label: 'Where used in ML?' }, { id: 'how', label: 'How it works internally' }, { id: 'basic', label: 'Basic Usage' }, { id: 'what-lense', label: 'what_lense' }, { id: 'how-lense', label: 'how_lense' }, { id: 'params', label: 'Parameters' }, { id: 'returns', label: 'Returns: LinalgResult' }];

export default function MatmulPage() {
    return (
        <>
            <Head><title>matmul — ML-Lens math.linalg</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'math.linalg', href: '/guide/linalg' }, { label: 'matmul', href: '/guide/linalg/matmul' }]}
                toc={toc}
                prev={{ label: 'math.linalg Overview', href: '/guide/linalg' }}
                next={{ label: 'solve', href: '/guide/linalg/solve' }}
            >
                <h1>matmul</h1>
                <blockquote>Matrix multiplication — the backbone of neural networks, linear regression, and virtually all ML computation.</blockquote>

                <h2 id="what">What is matmul?</h2>
                <p>Matrix multiplication computes the dot product between rows of matrix A and columns of matrix B. Each element <code className="inline-code">C[i,j] = Σₖ A[i,k] · B[k,j]</code>.</p>
                <p>Supported shapes:</p>
                <ul>
                    <li><strong>1D × 1D</strong> → scalar (inner product)</li>
                    <li><strong>2D × 1D</strong> → 1D vector</li>
                    <li><strong>2D × 2D</strong> → 2D matrix</li>
                </ul>
                <p>Shape rule: A must be (m×k) and B must be (k×n) — the inner dimensions must match.</p>

                <h2 id="ml">Where used in ML?</h2>
                <ul>
                    <li><strong>Dense/Linear Layers:</strong> output = Weights @ inputs + bias</li>
                    <li><strong>Attention mechanisms:</strong> QKᵀ in self-attention</li>
                    <li><strong>Covariance matrices:</strong> XᵀX in PCA, regression</li>
                    <li><strong>Convolutions:</strong> often lowered to matmul via im2col</li>
                    <li><strong>Batch inference:</strong> entire dataset through model simultaneously</li>
                </ul>

                <h2 id="how">How it works internally</h2>
                <p>mllense uses an <strong>algorithm registry</strong> that auto-selects the best implementation:</p>
                <ul>
                    <li><strong>Small matrices:</strong> Naive triple-loop — O(m·k·n), readable</li>
                    <li><strong>Medium matrices:</strong> Block multiplication — cache-optimized chunks</li>
                    <li><strong>Large matrices / numpy backend:</strong> <code className="inline-code">np.matmul</code> → BLAS routines</li>
                </ul>

                <h2 id="basic">Basic Usage</h2>
                <CodeBlock code={`from mllense.math.linalg import matmul

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

result = matmul(A, B)
print(result)              # → [[19, 22], [43, 50]]
print(result.value)        # ndarray access

# NumPy arrays work too
import numpy as np
A_np = np.array([[1, 2], [3, 4]])
result = matmul(A_np, B)   # Returns ndarray if input is ndarray`} />

                <h2 id="what-lense">With what_lense</h2>
                <CodeBlock code={`result = matmul(A, B, what_lense=True)
print(result.what_lense)
# → "=== WHAT: Matrix Multiplication ===
#    Matrix multiplication (dot product) combines the rows of the first
#    matrix with the columns of the second. Each element in the result
#    is the sum of the products of corresponding elements.
#
#    === WHY we need it in ML ===
#    It allows us to compute many linear combinations at once. It forms
#    the core of feed-forward neural networks (Weights * Inputs),
#    attention mechanisms (Q * K^T), and embedding projections."`} />

                <h2 id="how-lense">With how_lense</h2>
                <CodeBlock code={`result = matmul(A, B, how_lense=True)
print(result.how_lense)
# → "1. Validated shapes: A(2x2) @ B(2x2) -> Result(2x2).
#    2. Initialized empty output matrix of shape 2x2.
#    3. Triple-loop computation (m=2, k=2, n=2):
#       - Result[0][0] += A[0][0] * B[0][0] (1 * 5 = 5)
#       - Result[0][0] += A[0][1] * B[1][0] (2 * 7 = 14)
#       ... (subsequent operations)
#    4. Finished matrix multiplication."`} />

                <h2 id="params">Parameters</h2>
                <table>
                    <thead><tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">a</code></td><td>MatrixLike | VectorLike</td><td>—</td><td>Left operand</td></tr>
                        <tr><td><code className="inline-code">b</code></td><td>MatrixLike | VectorLike</td><td>—</td><td>Right operand</td></tr>
                        <tr><td><code className="inline-code">backend</code></td><td>str | None</td><td>None</td><td>Override backend: "numpy", "python"</td></tr>
                        <tr><td><code className="inline-code">mode</code></td><td>str | None</td><td>None</td><td>Override mode: "fast", "educational", "debug"</td></tr>
                        <tr><td><code className="inline-code">algorithm</code></td><td>str | None</td><td>None</td><td>Force algorithm: "naive", "block", "strassen"</td></tr>
                        <tr><td><code className="inline-code">trace_enabled</code></td><td>bool | None</td><td>None</td><td>Override global trace flag</td></tr>
                        <tr><td><code className="inline-code">what_lense</code></td><td>bool</td><td>True</td><td>Attach theoretical explanation</td></tr>
                        <tr><td><code className="inline-code">how_lense</code></td><td>bool</td><td>False</td><td>Attach step-by-step operational trace</td></tr>
                    </tbody>
                </table>

                <h2 id="returns">Returns: LinalgResult</h2>
                <table>
                    <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">value</code></td><td>ndarray | list | float</td><td>The result matrix, vector, or scalar</td></tr>
                        <tr><td><code className="inline-code">what_lense</code></td><td>str</td><td>Theoretical context (if enabled)</td></tr>
                        <tr><td><code className="inline-code">how_lense</code></td><td>str</td><td>Step-by-step computation trace (if enabled)</td></tr>
                        <tr><td><code className="inline-code">algorithm_used</code></td><td>str</td><td>Which algorithm was selected (e.g. "naive_matmul")</td></tr>
                        <tr><td><code className="inline-code">complexity</code></td><td>str</td><td>Big-O complexity (e.g. "O(m*k*n)")</td></tr>
                    </tbody>
                </table>
            </GuideLayout>
        </>
    );
}
