import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

export default function RandomForestRegressorPage() {
    return (
        <>
            <Head><title>RandomForestRegressor — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'Models', href: '/guide/models' }, { label: 'RandomForestRegressor', href: '/guide/models/random-forest-regressor' }]}
                toc={[{ id: 'what', label: 'What is it?' }, { id: 'usage', label: 'Usage' }, { id: 'params', label: 'Parameters' }]}
                prev={{ label: 'RandomForestClassifier', href: '/guide/models/random-forest-classifier' }}
                next={{ label: 'KMeans', href: '/guide/models/kmeans' }}
            >
                <h1>RandomForestRegressor</h1>
                <blockquote>Bagged ensemble of regression trees — each tree predicts a value and the forest returns their mean.</blockquote>

                <h2 id="what">What is it?</h2>
                <p>The regression counterpart to <code className="inline-code">RandomForestClassifier</code>. Instead of majority voting, predictions from each tree are <strong>averaged</strong> (mean) to produce the final continuous output.</p>
                <p>This reduces variance (overfitting) inherent in a single regression tree without significantly increasing bias.</p>

                <h2 id="usage">Usage</h2>
                <CodeBlock code={`from mllense.models import RandomForestRegressor

model = RandomForestRegressor(n_estimators=200, max_depth=8, what_lense=True)
model.fit(X_train, y_train)
result = model.predict(X_test)
print(result)             # Continuous predictions
print(model.score(X_test, y_test))  # R² score`} />

                <h2 id="params">Parameters</h2>
                <table>
                    <thead><tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">n_estimators</code></td><td>int</td><td>100</td><td>Number of regression trees</td></tr>
                        <tr><td><code className="inline-code">max_depth</code></td><td>int | None</td><td>None</td><td>Max depth of individual trees</td></tr>
                        <tr><td><code className="inline-code">what_lense</code></td><td>bool</td><td>False</td><td>Enable theoretical context</td></tr>
                        <tr><td><code className="inline-code">how_lense</code></td><td>bool</td><td>False</td><td>Enable aggregation trace</td></tr>
                    </tbody>
                </table>
            </GuideLayout>
        </>
    );
}
