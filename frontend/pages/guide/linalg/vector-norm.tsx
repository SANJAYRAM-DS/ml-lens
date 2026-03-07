import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

export default function NormsPage() {
    return (
        <>
            <Head><title>Norms & Diagnostics — ML-Lens math.linalg</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'math.linalg', href: '/guide/linalg' }, { label: 'Norms & Diagnostics', href: '/guide/linalg/vector-norm' }]}
                toc={[{ id: 'vector', label: 'vector_norm' }, { id: 'frobenius', label: 'frobenius_norm' }, { id: 'spectral', label: 'spectral_norm' }, { id: 'condition', label: 'condition_number' }, { id: 'rank', label: 'matrix_rank' }, { id: 'stability', label: 'stability_report' }]}
                prev={{ label: 'svd', href: '/guide/linalg/svd' }}
                next={{ label: 'Lenses Guide', href: '/guide/lenses' }}
            >
                <h1>Norms & Diagnostics</h1>
                <p>Tools for measuring matrix magnitudes and assessing numerical stability of computations.</p>

                <h2 id="vector">vector_norm</h2>
                <p>Computes the p-norm of a 1D vector. L1, L2 (Euclidean), and L-infinity norms supported.</p>
                <CodeBlock code={`from mllense.math.linalg import vector_norm
v = [3, 4]
print(vector_norm(v, ord=2))       # → 5.0 (Euclidean: √(9+16))
print(vector_norm(v, ord=1))       # → 7.0 (Manhattan: 3+4)
print(vector_norm(v, ord=float('inf')))  # → 4.0 (Max: max(3,4))`} />

                <h2 id="frobenius">frobenius_norm</h2>
                <p>The Frobenius norm: square root of the sum of all squared elements. The matrix equivalent of the L2 vector norm.</p>
                <CodeBlock code={`from mllense.math.linalg import frobenius_norm
A = [[1, 2], [3, 4]]
f = frobenius_norm(A)
# √(1² + 2² + 3² + 4²) = √30 ≈ 5.477
print(f)`} />

                <h2 id="spectral">spectral_norm</h2>
                <p>The 2-norm (spectral norm) — the largest singular value of the matrix. Measures the maximum amplification factor of the transformation.</p>
                <CodeBlock code={`from mllense.math.linalg import spectral_norm
A = [[1, 2], [3, 4]]
s = spectral_norm(A)  # → ≈ 5.465 (largest singular value)`} />

                <h2 id="condition">condition_number</h2>
                <p>Ratio of largest to smallest singular value (σ_max / σ_min). A high condition number means the matrix is nearly singular and small input errors amplify greatly.</p>
                <CodeBlock code={`from mllense.math.linalg import condition_number
A = [[1, 2], [3, 4]]
cn = condition_number(A)
print(cn)  # Low = well-conditioned, High = ill-conditioned (>1000 = problematic)`} />

                <h2 id="rank">matrix_rank</h2>
                <p>Determines the rank (number of linearly independent rows/columns) of a matrix using SVD with a numerical threshold.</p>
                <CodeBlock code={`from mllense.math.linalg import matrix_rank
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Rank deficient (3rd row = 2*2nd - 1st)
print(matrix_rank(A))  # → 2`} />

                <h2 id="stability">stability_report / full_diagnostic_report</h2>
                <p>High-level diagnostic summaries combining rank, condition number, and perturbation analysis.</p>
                <CodeBlock code={`from mllense.math.linalg import stability_report, full_diagnostic_report

A = [[1, 2], [3, 4]]
sr = stability_report(A)         # Perturbation analysis report
fdr = full_diagnostic_report(A)  # All: rank, cond, stability combined

print(sr)   # Stability assessment string
print(fdr)  # Full combined diagnostic report`} />
            </GuideLayout>
        </>
    );
}
