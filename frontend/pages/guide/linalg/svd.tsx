import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

export default function SVDPage() {
    return (
        <>
            <Head><title>svd — ML-Lens math.linalg</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'math.linalg', href: '/guide/linalg' }, { label: 'svd', href: '/guide/linalg/svd' }]}
                toc={[{ id: 'what', label: 'What is SVD?' }, { id: 'ml', label: 'Where in ML?' }, { id: 'usage', label: 'Usage' }, { id: 'params', label: 'Parameters' }, { id: 'returns', label: 'Returns' }]}
                prev={{ label: 'qr', href: '/guide/linalg/qr' }}
                next={{ label: 'eig', href: '/guide/linalg/eig' }}
            >
                <h1>svd</h1>
                <blockquote>Singular Value Decomposition — factorizes any matrix into U Σ Vᵀ, revealing its fundamental geometric transformation.</blockquote>

                <h2 id="what">What is SVD?</h2>
                <p>SVD factorizes any m×n matrix A into three matrices: <code className="inline-code">A = U Σ Vᵀ</code></p>
                <ul>
                    <li><strong>U</strong> (m×m): Left singular vectors — orthonormal basis for column space</li>
                    <li><strong>Σ</strong> (m×n): Diagonal matrix of singular values — how much each direction is "stretched"</li>
                    <li><strong>Vᵀ</strong> (n×n): Right singular vectors — orthonormal basis for row space</li>
                </ul>
                <p>The singular values σ₁ ≥ σ₂ ≥ ... ≥ 0 reveal the "importance" of each dimension.</p>

                <h2 id="ml">Where used in ML?</h2>
                <ul>
                    <li><strong>PCA:</strong> Principal components = right singular vectors of centered X</li>
                    <li><strong>LSA/Topic Modeling:</strong> Truncated SVD (top-k singular values) for dimensionality reduction</li>
                    <li><strong>Recommender Systems:</strong> Matrix factorization via SVD (collaborative filtering)</li>
                    <li><strong>Pseudoinverse:</strong> A⁺ = V Σ⁺ Uᵀ for solving overdetermined systems</li>
                    <li><strong>Condition number:</strong> σ_max / σ_min measures numerical stability</li>
                    <li><strong>Low-rank approximation:</strong> Image compression</li>
                </ul>

                <h2 id="usage">Usage</h2>
                <CodeBlock code={`from mllense.math.linalg import svd

A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

U, S, Vt = svd(A, what_lense=True, how_lense=True)
print(U.shape)   # (3, 3) — left singular vectors
print(S)         # [1D array of singular values, descending]
print(Vt.shape)  # (3, 3) — right singular vectors (transposed)

# Reconstruct original matrix
import numpy as np
A_reconstructed = U @ np.diag(S) @ Vt

# Low-rank approximation (keep top 2 components)
k = 2
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]`} />

                <h2 id="params">Parameters</h2>
                <table>
                    <thead><tr><th>Parameter</th><th>Type</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">a</code></td><td>MatrixLike</td><td>Input matrix (any shape m×n)</td></tr>
                        <tr><td><code className="inline-code">what_lense</code></td><td>bool</td><td>Enable theoretical explanation</td></tr>
                        <tr><td><code className="inline-code">how_lense</code></td><td>bool</td><td>Enable decomposition trace</td></tr>
                    </tbody>
                </table>

                <h2 id="returns">Returns</h2>
                <table>
                    <thead><tr><th>Value</th><th>Shape</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">U</code></td><td>(m, m)</td><td>Left singular vectors (orthogonal)</td></tr>
                        <tr><td><code className="inline-code">S</code></td><td>(min(m,n),)</td><td>Singular values in descending order</td></tr>
                        <tr><td><code className="inline-code">Vt</code></td><td>(n, n)</td><td>Right singular vectors (transposed)</td></tr>
                    </tbody>
                </table>
            </GuideLayout>
        </>
    );
}
