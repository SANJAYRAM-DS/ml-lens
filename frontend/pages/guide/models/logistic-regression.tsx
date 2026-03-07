import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

const toc = [{ id: 'what', label: 'What is it?' }, { id: 'ml', label: 'Where used in ML?' }, { id: 'how', label: 'How it works' }, { id: 'usage', label: 'Usage' }, { id: 'lenses', label: 'Lenses' }, { id: 'params', label: 'Parameters' }, { id: 'returns', label: 'Returns' }];

export default function LogisticRegressionPage() {
    return (
        <>
            <Head><title>LogisticRegression — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'Models', href: '/guide/models' }, { label: 'LogisticRegression', href: '/guide/models/logistic-regression' }]}
                toc={toc}
                prev={{ label: 'LinearRegression', href: '/guide/models/linear-regression' }}
                next={{ label: 'DecisionTreeClassifier', href: '/guide/models/decision-tree-classifier' }}
            >
                <h1>LogisticRegression</h1>
                <blockquote>Binary classification via gradient descent — maps linear outputs to probabilities using the sigmoid function.</blockquote>

                <h2 id="what">What is it?</h2>
                <p>Logistic Regression is a <strong>classification</strong> algorithm that predicts the probability of a binary outcome. Despite the name, it is not a regression algorithm — it uses a logistic (sigmoid) function to squash the linear output into [0, 1].</p>
                <p>The sigmoid: <code className="inline-code">σ(z) = 1 / (1 + e⁻ᶻ)</code> where <code className="inline-code">z = Xw + b</code></p>

                <h2 id="ml">Where used in ML?</h2>
                <ul>
                    <li>Spam detection, disease diagnosis, click-through rate prediction</li>
                    <li>The output layer of binary neural networks</li>
                    <li>Baseline for any binary classification problem</li>
                    <li>Feature importance ranking via coefficient magnitudes</li>
                </ul>

                <h2 id="how">How it works internally</h2>
                <ol>
                    <li>Initialize weights w=0, bias b=0</li>
                    <li>For each iteration: compute z = X·w + b, then ŷ = σ(z)</li>
                    <li>Compute gradient: dw = (1/m)·Xᵀ·(ŷ−y), db = (1/m)·Σ(ŷ−y)</li>
                    <li>Update: w = w − lr·dw, b = b − lr·db</li>
                    <li>Repeat until max_iter, then threshold at 0.5 to classify</li>
                </ol>
                <p><strong>Complexity:</strong> O(max_iter × n_features × n_samples)</p>

                <h2 id="usage">Basic Usage</h2>
                <CodeBlock code={`from mllense.models import LogisticRegression

model = LogisticRegression(learning_rate=0.01, max_iter=1000)
model.fit(X_train, y_train)

result = model.predict(X_test)
print(result)        # 0/1 class labels

proba = model.predict_proba(X_test)
print(proba)         # [[p0, p1], ...] probabilities`} />

                <h2 id="lenses">Using the Lenses</h2>
                <CodeBlock code={`model = LogisticRegression(what_lense=True, how_lense=True)
model.fit(X, y)
result = model.predict(X)
print(result.what_lense)
# → "=== WHAT: logistic_regression ===
#    Logistic Regression via gradient descent mapping linear outputs..."
print(result.how_lense)
# → "1. Initializing Gradient Descent (iter=1000, lr=0.01)
#    2. Entering optimization loop: w = w - lr * dJ/dw..."`} />

                <h2 id="params">Parameters</h2>
                <table>
                    <thead><tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">learning_rate</code></td><td>float</td><td>0.01</td><td>Step size for gradient descent</td></tr>
                        <tr><td><code className="inline-code">max_iter</code></td><td>int</td><td>1000</td><td>Maximum gradient descent iterations</td></tr>
                        <tr><td><code className="inline-code">what_lense</code></td><td>bool</td><td>False</td><td>Attach theoretical explanation to results</td></tr>
                        <tr><td><code className="inline-code">how_lense</code></td><td>bool</td><td>False</td><td>Attach step-by-step trace to results</td></tr>
                    </tbody>
                </table>

                <h2 id="returns">Returns: ModelResult</h2>
                <table>
                    <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">value</code></td><td>ndarray[int]</td><td>Binary class predictions (0 or 1)</td></tr>
                        <tr><td><code className="inline-code">what_lense</code></td><td>str</td><td>Algorithm theory explanation</td></tr>
                        <tr><td><code className="inline-code">how_lense</code></td><td>str</td><td>Gradient descent iteration trace</td></tr>
                        <tr><td><code className="inline-code">metadata.complexity</code></td><td>str</td><td>O(max_iter × n_features × n_samples)</td></tr>
                    </tbody>
                </table>
            </GuideLayout>
        </>
    );
}
