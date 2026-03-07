import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

export default function DecisionTreeRegressorPage() {
    return (
        <>
            <Head><title>DecisionTreeRegressor — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'Models', href: '/guide/models' }, { label: 'DecisionTreeRegressor', href: '/guide/models/decision-tree-regressor' }]}
                toc={[{ id: 'what', label: 'What is it?' }, { id: 'usage', label: 'Usage' }, { id: 'params', label: 'Parameters' }]}
                prev={{ label: 'DecisionTreeClassifier', href: '/guide/models/decision-tree-classifier' }}
                next={{ label: 'RandomForestClassifier', href: '/guide/models/random-forest-classifier' }}
            >
                <h1>DecisionTreeRegressor</h1>
                <blockquote>CART decision tree for regression — recursively splits to minimize variance (MSE).</blockquote>

                <h2 id="what">What is it?</h2>
                <p>The regressor variant of the decision tree. Instead of class labels at leaf nodes, it stores the <strong>mean target value</strong> of all training samples that reached that leaf. Splits are chosen to minimize mean squared error in each partition.</p>
                <p>The prediction for a new sample is the mean y-value of the training points in its leaf node.</p>

                <h2 id="ml">Where used in ML?</h2>
                <ul>
                    <li>Non-linear regression without feature engineering</li>
                    <li>Gradient Boosted Trees (base learner)</li>
                    <li>Piecewise constant function approximation</li>
                </ul>

                <h2 id="usage">Usage</h2>
                <CodeBlock code={`from mllense.models import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=4, how_lense=True)
model.fit(X_train, y_train)
result = model.predict(X_test)
print(result)
print(result.how_lense)
# → "1. Fitting Decision Tree Regressor (max_depth=4)
#    2. Recursively splitting data to minimize MSE..."
print(model.score(X_test, y_test))  # R² score`} />

                <h2 id="params">Parameters</h2>
                <table>
                    <thead><tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">max_depth</code></td><td>int | None</td><td>None</td><td>Maximum depth of tree growth</td></tr>
                        <tr><td><code className="inline-code">what_lense</code></td><td>bool</td><td>False</td><td>Enable theoretical explanations</td></tr>
                        <tr><td><code className="inline-code">how_lense</code></td><td>bool</td><td>False</td><td>Enable step-by-step trace</td></tr>
                    </tbody>
                </table>
            </GuideLayout>
        </>
    );
}
