import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

const toc = [{ id: 'what', label: 'What is it?' }, { id: 'ml', label: 'Where used in ML?' }, { id: 'how', label: 'How it works' }, { id: 'usage', label: 'Usage' }, { id: 'lenses', label: 'Lenses' }, { id: 'params', label: 'Parameters' }, { id: 'returns', label: 'Returns' }];

export default function LinearRegressionPage() {
    return (
        <>
            <Head><title>LinearRegression — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'Models', href: '/guide/models' }, { label: 'LinearRegression', href: '/guide/models/linear-regression' }]}
                toc={toc}
                prev={{ label: 'Models Overview', href: '/guide/models' }}
                next={{ label: 'LogisticRegression', href: '/guide/models/logistic-regression' }}
            >
                <h1>LinearRegression</h1>
                <blockquote>Ordinary least squares linear regression — the foundational algorithm for supervised numeric prediction.</blockquote>

                <h2 id="what">What is it?</h2>
                <p>Linear Regression models the relationship between one or more input features and a continuous output by fitting a linear equation. Given matrix X (features) and vector y (targets), it finds coefficients θ such that <strong>y ≈ Xθ</strong>.</p>
                <p>The closed-form solution uses the <strong>Normal Equation</strong>: <code className="inline-code">θ = (Xᵀ X)⁻¹ Xᵀ y</code></p>

                <h2 id="ml">Where is it used in ML?</h2>
                <ul>
                    <li>House price prediction, stock forecasting, demand estimation</li>
                    <li>As a baseline model before applying complex algorithms</li>
                    <li>Underlying mechanism in ridge/lasso regression (with regularization)</li>
                    <li>Understanding the optimization landscape of neural networks</li>
                </ul>

                <h2 id="how">How it works internally</h2>
                <p>mllense's LinearRegression solves the normal equation directly:</p>
                <ol>
                    <li>Appends a column of ones to X to compute the bias/intercept</li>
                    <li>Computes <code className="inline-code">Xᵀ @ X</code> — a (p+1)×(p+1) correlation matrix</li>
                    <li>Computes its inverse via <code className="inline-code">np.linalg.inv</code></li>
                    <li>Multiplies by <code className="inline-code">Xᵀ @ y</code> to get the optimal coefficients θ</li>
                    <li>Splits θ into <code className="inline-code">intercept_</code> and <code className="inline-code">coef_</code></li>
                </ol>
                <p><strong>Complexity:</strong> O(n·p² + p³) where n=samples, p=features</p>

                <h2 id="usage">Basic Usage</h2>
                <CodeBlock code={`import numpy as np
from mllense.models import LinearRegression

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([2.1, 3.2, 4.0, 5.1])

model = LinearRegression(fit_intercept=True)
model.fit(X, y)
result = model.predict(X)
print(result)           # ModelResult — prints predicted values
print(result.value)     # Raw ndarray of predictions
print(model.coef_)      # Learned coefficients
print(model.intercept_) # Learned intercept`} />

                <h2 id="lenses">Using the Lenses</h2>
                <h3>what_lense — Theoretical Context</h3>
                <CodeBlock code={`model = LinearRegression(what_lense=True)
model.fit(X, y)
result = model.predict(X)
print(result.what_lense)
# → "=== WHAT: ordinary_least_squares ===
#    Linear Regression using Orthogonal/Singular/Normal Equation..."`} />
                <h3>how_lense — Operational Trace</h3>
                <CodeBlock code={`model = LinearRegression(how_lense=True)
model.fit(X, y)
result = model.predict(X)
print(result.how_lense)
# → "1. Initializing Linear Regression (fit_intercept=True)
#    2. Adding a column of ones to X for intercept term
#    3. Solving normal equations: theta = (X^T @ X)^-1 @ X^T @ y
#    4. Model fitted. Intercept: 0.1234"`} />

                <h2 id="params">Parameters</h2>
                <table>
                    <thead><tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">fit_intercept</code></td><td>bool</td><td>True</td><td>Whether to fit a bias/intercept term</td></tr>
                        <tr><td><code className="inline-code">what_lense</code></td><td>bool</td><td>False</td><td>Attach theoretical explanation to results</td></tr>
                        <tr><td><code className="inline-code">how_lense</code></td><td>bool</td><td>False</td><td>Attach step-by-step trace to results</td></tr>
                    </tbody>
                </table>

                <h2 id="returns">Returns: ModelResult</h2>
                <table>
                    <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">value</code></td><td>ndarray</td><td>Predicted continuous values</td></tr>
                        <tr><td><code className="inline-code">what_lense</code></td><td>str</td><td>Algorithm theory explanation</td></tr>
                        <tr><td><code className="inline-code">how_lense</code></td><td>str</td><td>Step-by-step execution log</td></tr>
                        <tr><td><code className="inline-code">metadata.complexity</code></td><td>str</td><td>O(n·p² + p³)</td></tr>
                    </tbody>
                </table>
            </GuideLayout>
        </>
    );
}
