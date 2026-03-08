import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

const toc = [
    { id: 'linear', label: 'LinearRegression' },
    { id: 'logistic', label: 'LogisticRegression' },
    { id: 'tree', label: 'DecisionTree' },
    { id: 'forest', label: 'RandomForest' },
    { id: 'kmeans', label: 'KMeans' },
    { id: 'base', label: 'BaseEstimator' },
    { id: 'result', label: 'ModelResult' },
];

export default function ModelsGuide() {
    return (
        <>
            <Head><title>Models API — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Home', href: '/' }, { label: 'Guide', href: '/guide' }, { label: 'Models', href: '/guide/models' }]}
                toc={toc}
                prev={{ label: 'User Guide', href: '/guide' }}
                next={{ label: 'LinearRegression', href: '/guide/models/linear-regression' }}
            >
                <h1>Models API</h1>
                <p>All models in <code className="inline-code">mllense.models</code> follow the Scikit-Learn interface with additional educational tracing via <code className="inline-code">what_lense</code> and <code className="inline-code">how_lense</code>.</p>

                <h2 id="linear">LinearRegression</h2>
                <p>Ordinary least squares via the normal equation. Returns a <code className="inline-code">ModelResult</code> from <code className="inline-code">predict()</code>.</p>
                <CodeBlock code={`from mllense.models import LinearRegression\nmodel = LinearRegression(fit_intercept=True, what_lense=True, how_lense=True)\nmodel.fit(X, y)\nresult = model.predict(X)`} />

                <h2 id="logistic">LogisticRegression</h2>
                <p>Binary classification using gradient descent + sigmoid activation.</p>
                <CodeBlock code={`from mllense.models import LogisticRegression\nmodel = LogisticRegression(learning_rate=0.01, max_iter=1000, what_lense=True)\nmodel.fit(X, y)\nresult = model.predict(X)  # Returns ModelResult with 0/1 class labels`} />

                <h2 id="tree">DecisionTree</h2>
                <p>CART-style tree splitting. Both classifier and regressor variants available.</p>
                <CodeBlock code={`from mllense.models import DecisionTreeClassifier, DecisionTreeRegressor\nclf = DecisionTreeClassifier(max_depth=5, how_lense=True)\nreg = DecisionTreeRegressor(max_depth=5)`} />

                <h2 id="forest">RandomForest</h2>
                <p>Bagged ensemble of decision trees. N estimators each trained on a bootstrapped sample.</p>
                <CodeBlock code={`from mllense.models import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100, max_depth=10, what_lense=True)\nmodel.fit(X, y)\nresult = model.predict(X)`} />

                <h2 id="kmeans">KMeans</h2>
                <p>Unsupervised clustering. Iteratively assigns points to the nearest centroid until convergence.</p>
                <CodeBlock code={`from mllense.models import KMeans\nmodel = KMeans(n_clusters=3, random_state=42, how_lense=True)\nmodel.fit(X)\nprint(model.labels_)\nprint(model.cluster_centers_)`} />

                <h2 id="base">BaseEstimator</h2>
                <p>All models inherit from <code className="inline-code">mllense.models.base.BaseEstimator</code>. It provides:</p>
                <ul>
                    <li><code className="inline-code">what_lense_enabled</code> — flag to enable theoretical explanations</li>
                    <li><code className="inline-code">how_lense_enabled</code> — flag to enable step-by-step tracing</li>
                    <li><code className="inline-code">context</code> — an <code className="inline-code">ExecutionContext</code> object routing configuration</li>
                    <li><code className="inline-code">_generate_what_lense()</code> — builds the what explanation from metadata</li>
                    <li><code className="inline-code">_finalize_how_lense(trace)</code> — finalizes the trace steps log</li>
                </ul>

                <h2 id="result">ModelResult</h2>
                <p>The return type of all <code className="inline-code">predict()</code> methods. Behaves like an ndarray but has extra properties:</p>
                <table>
                    <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">value</code></td><td>ndarray</td><td>The actual prediction array</td></tr>
                        <tr><td><code className="inline-code">what_lense</code></td><td>str</td><td>Theoretical context string (if enabled)</td></tr>
                        <tr><td><code className="inline-code">how_lense</code></td><td>str</td><td>Step-by-step operational trace (if enabled)</td></tr>
                        <tr><td><code className="inline-code">metadata</code></td><td>ModelMetadata</td><td>Name, type, complexity of the algorithm</td></tr>
                        <tr><td><code className="inline-code">algorithm_used</code></td><td>str</td><td>Human-readable algorithm name</td></tr>
                    </tbody>
                </table>
            </GuideLayout>
        </>
    );
}
