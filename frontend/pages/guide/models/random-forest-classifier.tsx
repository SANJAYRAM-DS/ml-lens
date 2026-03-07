import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

export default function RandomForestClassifierPage() {
    return (
        <>
            <Head><title>RandomForestClassifier — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'Models', href: '/guide/models' }, { label: 'RandomForestClassifier', href: '/guide/models/random-forest-classifier' }]}
                toc={[{ id: 'what', label: 'What is it?' }, { id: 'ml', label: 'Where used in ML?' }, { id: 'usage', label: 'Usage' }, { id: 'params', label: 'Parameters' }]}
                prev={{ label: 'DecisionTreeRegressor', href: '/guide/models/decision-tree-regressor' }}
                next={{ label: 'RandomForestRegressor', href: '/guide/models/random-forest-regressor' }}
            >
                <h1>RandomForestClassifier</h1>
                <blockquote>Ensemble of decision trees via bagging — each tree trained on a bootstrapped sample with majority vote aggregation.</blockquote>

                <h2 id="what">What is it?</h2>
                <p>Random Forest reduces the variance of a single decision tree by averaging many trees grown on different subsets of the training data. Each tree is a full decision tree trained on a <strong>bootstrapped sample</strong> (random sampling with replacement). Predictions are aggregated by <strong>majority vote</strong> across all trees.</p>
                <p><strong>Key insight:</strong> Individual trees may overfit, but averaging n=100+ uncorrelated trees produces a robust, generalizing ensemble.</p>

                <h2 id="ml">Where used in ML?</h2>
                <ul>
                    <li>Tabular data classification (Kaggle competitions, banking, healthcare)</li>
                    <li>Feature importance estimation (by measuring impurity reduction)</li>
                    <li>Robust baselines — often beats complex models on structured data</li>
                    <li>When you need good OOB (out-of-bag) error estimates without a dedicated test set</li>
                </ul>

                <h2 id="usage">Usage</h2>
                <CodeBlock code={`from mllense.models import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    what_lense=True,
    how_lense=True
)
model.fit(X_train, y_train)
result = model.predict(X_test)

print(result)
print(result.how_lense)
# → "1. Fitting Random Forest Classifier with M=100 estimators
#    2. Creating bootstrapped dataset and growing tree 1
#    3. Creating bootstrapped dataset and growing tree 2
#    ... Skipping subsequent tree iteration logs ...
#    N. Forest constructed successfully"`} />

                <h2 id="params">Parameters</h2>
                <table>
                    <thead><tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">n_estimators</code></td><td>int</td><td>100</td><td>Number of trees in the forest</td></tr>
                        <tr><td><code className="inline-code">max_depth</code></td><td>int | None</td><td>None</td><td>Max depth of individual trees</td></tr>
                        <tr><td><code className="inline-code">what_lense</code></td><td>bool</td><td>False</td><td>Enable theoretical context</td></tr>
                        <tr><td><code className="inline-code">how_lense</code></td><td>bool</td><td>False</td><td>Enable bagging trace</td></tr>
                    </tbody>
                </table>
            </GuideLayout>
        </>
    );
}
