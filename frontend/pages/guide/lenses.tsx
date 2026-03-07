import Head from 'next/head';
import GuideLayout from '../../components/GuideLayout';
import CodeBlock from '../../components/CodeBlock';

const toc = [{ id: 'what-lense', label: 'what_lense' }, { id: 'how-lense', label: 'how_lense' }, { id: 'per-call', label: 'Per-call usage' }, { id: 'global', label: 'Global usage' }, { id: 'result', label: 'Result object' }, { id: 'examples', label: 'Examples' }];

export default function LensesPage() {
    return (
        <>
            <Head><title>Lenses — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'Lenses', href: '/guide/lenses' }]}
                toc={toc}
                prev={{ label: 'math.linalg Overview', href: '/guide/linalg' }}
                next={{ label: 'GlobalConfig', href: '/guide/config' }}
            >
                <h1>what_lense & how_lense</h1>
                <blockquote>The educational observation layer of mllense — trace every computation, understand every step.</blockquote>

                <p>Every function in both <code className="inline-code">mllense.models</code> and <code className="inline-code">mllense.math.linalg</code> supports two tracing parameters:</p>

                <h2 id="what-lense">what_lense — Theoretical Context</h2>
                <p>When enabled, every result object carries a <code className="inline-code">what_lense</code> string that explains:</p>
                <ul>
                    <li>🧠 <strong>WHAT</strong> the algorithm is mathematically</li>
                    <li>💡 <strong>WHY</strong> it exists and what problem it solves</li>
                    <li>🏭 <strong>WHERE</strong> it's used in real ML pipelines</li>
                </ul>
                <CodeBlock code={`from mllense.math.linalg import matmul

result = matmul([[1,2],[3,4]], [[5,6],[7,8]], what_lense=True)
print(result.what_lense)
# → "=== WHAT: Matrix Multiplication ===
#    Matrix multiplication (dot product) combines the rows of the first
#    matrix with the columns of the second. Each element in the result
#    is the sum of the products of corresponding elements.
#
#    === WHY we need it in ML ===
#    It allows us to compute many linear combinations at once. It forms
#    the core of feed-forward neural networks (Weights * Inputs),
#    attention mechanisms (Q * K^T), and embedding projections.
#    
#    === WHERE it is used in Real ML ===
#    1. Dense/Linear Layers: Output = Weights @ Inputs + Bias.
#    2. Convolutions: Often lowered to matrix multiplication (im2col).
#    3. Transformers: Self-attention relies heavily on batched matmuls."`} />

                <h2 id="how-lense">how_lense — Operational Trace</h2>
                <p>When enabled, the result carries a <code className="inline-code">how_lense</code> string showing exactly what happened computationally:</p>
                <ul>
                    <li>🔢 Shape checks and validations performed</li>
                    <li>➕ Individual arithmetic operations (for small matrices)</li>
                    <li>🔄 Loop iterations, pivot swaps, centroid movements</li>
                    <li>✅ Convergence / completion conditions</li>
                </ul>
                <CodeBlock code={`result = matmul([[1,2],[3,4]], [[5,6],[7,8]], how_lense=True)
print(result.how_lense)
# → "1. Validated shapes: A(2x2) @ B(2x2) -> Result(2x2).
#    2. Initialized empty output matrix of shape 2x2.
#    3. Triple-loop computation (m=2, k=2, n=2):
#       - Result[0][0] += A[0][0] * B[0][0] (1 * 5 = 5)
#       - Result[0][0] += A[0][1] * B[1][0] (2 * 7 = 14)
#       - Result[0][1] += A[0][0] * B[0][1] (1 * 6 = 6)
#       - Result[0][1] += A[0][1] * B[1][1] (2 * 8 = 16)
#    4. Finished matrix multiplication."`} />

                <h2 id="per-call">Per-call usage (override)</h2>
                <p>Pass <code className="inline-code">what_lense</code> and <code className="inline-code">how_lense</code> directly to any function or model constructor:</p>
                <CodeBlock code={`# linalg functions
result = matmul(A, B, what_lense=True, how_lense=True)

# models
from mllense.models import LinearRegression
model = LinearRegression(what_lense=True, how_lense=True)
model.fit(X, y)
result = model.predict(X)

# Access lens output
print(result.what_lense)  # str
print(result.how_lense)   # str`} />

                <h2 id="global">Global usage (GlobalConfig)</h2>
                <CodeBlock code={`from mllense.math.linalg import GlobalConfig
GlobalConfig.trace_enabled = True  # Enables how_lense globally for all linalg ops
GlobalConfig.default_mode = "educational"  # Sets educational mode globally`} />

                <h2 id="result">The Result Object</h2>
                <p>All operations return a wrapper object that behaves <strong>exactly like the numeric value</strong> but carries trace strings:</p>
                <CodeBlock code={`result = matmul(A, B, what_lense=True)

# Acts like ordinary matrix
print(result)            # [[19.0, 22.0], [43.0, 50.0]]
print(result[0])         # [19.0, 22.0] — indexable
print(len(result))       # 2 — len() works
import numpy as np
arr = np.array(result.value)  # Get pure ndarray

# Educational data attached
print(result.what_lense)      # str
print(result.how_lense)       # str
print(result.algorithm_used)  # "naive_matmul"
print(result.complexity)      # "O(m*k*n)"`} />

                <h2 id="examples">More Examples</h2>
                <CodeBlock code={`from mllense.models import KMeans

model = KMeans(n_clusters=3, how_lense=True, what_lense=True)
model.fit(X)

# Access lens data after fit
print(model.what_lense)  # Attached to model after fit()
print(model.how_lense)   # Convergence iteration trace

# Predict also returns TracedResult
result = model.predict(X_new)
print(result.how_lense)  # Distance evaluation trace`} />
            </GuideLayout>
        </>
    );
}
