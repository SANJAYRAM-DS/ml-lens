import Head from 'next/head';
import GuideLayout from '../../components/GuideLayout';
import CodeBlock from '../../components/CodeBlock';

const toc = [
    { id: 'overview', label: 'Overview' },
    { id: 'models', label: 'Models' },
    { id: 'linalg', label: 'math.linalg' },
    { id: 'lenses', label: 'Lenses' },
    { id: 'config', label: 'GlobalConfig' },
];

export default function GuideIndex() {
    return (
        <>
            <Head><title>User Guide — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Home', href: '/' }, { label: 'User Guide', href: '/guide' }]}
                toc={toc}
                next={{ label: 'Models API', href: '/guide/models' }}
            >
                <h1 id="overview">User Guide</h1>
                <blockquote>mllense is a production-grade, educational ML and linear algebra framework that lets you <strong>trace every computation</strong> and <strong>understand every prediction</strong>.</blockquote>

                <h2 id="models">mllense.models</h2>
                <p>The <code className="inline-code">models</code> subpackage provides Scikit-Learn–compatible estimators with built-in tracing. Every model accepts <code className="inline-code">what_lense</code> and <code className="inline-code">how_lense</code> parameters.</p>
                <ul>
                    <li><strong>LinearRegression</strong> — OLS via normal equations</li>
                    <li><strong>LogisticRegression</strong> — Gradient descent + sigmoid</li>
                    <li><strong>DecisionTreeClassifier / Regressor</strong> — CART splitting</li>
                    <li><strong>RandomForestClassifier / Regressor</strong> — Bagged ensemble</li>
                    <li><strong>KMeans</strong> — Iterative Euclidean centroid assignment</li>
                </ul>
                <CodeBlock code={`from mllense.models import LinearRegression\nmodel = LinearRegression(what_lense=True, how_lense=True)\nmodel.fit(X, y)\nresult = model.predict(X)\nprint(result.what_lense)  # Theoretical explanation\nprint(result.how_lense)   # Step-by-step trace`} />

                <h2 id="linalg">mllense.math.linalg</h2>
                <p>A NumPy-compatible linear algebra engine with 27+ operations — all supporting tracing. Switch backends between <code className="inline-code">numpy</code>, <code className="inline-code">python</code>, or <code className="inline-code">numba</code>.</p>
                <CodeBlock code={`from mllense.math.linalg import matmul, svd, qr\nresult = matmul(A, B, what_lense=True, how_lense=True)\nprint(result.how_lense)`} />

                <h2 id="lenses">what_lense & how_lense</h2>
                <p>Every function returns a result object (<code className="inline-code">LinalgResult</code> or <code className="inline-code">ModelResult</code>) that wraps the numeric value AND attaches trace strings:</p>
                <ul>
                    <li><strong>what_lense</strong>: Theoretical context — what the algorithm does, why it exists, where it's used in ML</li>
                    <li><strong>how_lense</strong>: Operational trace — every step taken, shapes checked, pivots made, iterations run</li>
                </ul>

                <h2 id="config">GlobalConfig</h2>
                <p>Control backend, mode, and tracing defaults globally:</p>
                <CodeBlock code={`from mllense.math.linalg import GlobalConfig\nGlobalConfig.default_backend = "numpy"  # "python" | "numba"\nGlobalConfig.default_mode = "educational"  # "fast" | "debug"\nGlobalConfig.trace_enabled = True`} />
            </GuideLayout>
        </>
    );
}
