import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

const toc = [{ id: 'what', label: 'What is it?' }, { id: 'ml', label: 'Where used in ML?' }, { id: 'how', label: 'How it works' }, { id: 'usage', label: 'Usage' }, { id: 'params', label: 'Parameters' }, { id: 'returns', label: 'Returns' }];

export default function DecisionTreeClassifierPage() {
    return (
        <>
            <Head><title>DecisionTreeClassifier — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'Models', href: '/guide/models' }, { label: 'DecisionTreeClassifier', href: '/guide/models/decision-tree-classifier' }]}
                toc={toc}
                prev={{ label: 'LogisticRegression', href: '/guide/models/logistic-regression' }}
                next={{ label: 'DecisionTreeRegressor', href: '/guide/models/decision-tree-regressor' }}
            >
                <h1>DecisionTreeClassifier</h1>
                <blockquote>CART decision tree for classification — recursively partitions the feature space to maximize information gain.</blockquote>

                <h2 id="what">What is it?</h2>
                <p>A Decision Tree Classifier builds a tree structure where each internal node tests a feature condition (e.g., X[2] &gt; 0.5), and each leaf assigns a class label. It greedily selects the split that most reduces impurity (Gini or entropy).</p>

                <h2 id="ml">Where used in ML?</h2>
                <ul>
                    <li>Medical diagnosis, fraud detection, credit scoring</li>
                    <li>The base estimator inside Random Forests and Gradient Boosting</li>
                    <li>Feature selection via built-in importance scores</li>
                    <li>Any classification task requiring interpretability</li>
                </ul>

                <h2 id="how">How it works internally</h2>
                <ol>
                    <li>For each candidate feature and threshold: compute information gain or Gini reduction</li>
                    <li>Select the best split and partition the data</li>
                    <li>Repeat recursively on each partition until max_depth or pure leaf</li>
                    <li>Assign the majority class of each leaf node for prediction</li>
                </ol>
                <p><strong>Complexity (training):</strong> O(n_samples × n_features × log(n_samples))</p>

                <h2 id="usage">Usage</h2>
                <CodeBlock code={`from mllense.models import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, what_lense=True, how_lense=True)
model.fit(X_train, y_train)
result = model.predict(X_test)
print(result)           # ModelResult — prints predicted class labels
print(result.value)     # Raw ndarray of class predictions
print(result.how_lense)
# → "1. Fitting Decision Tree Classifier (max_depth=5)
#    2. Recursively splitting data to maximize information gain..."`} />

                <h2 id="params">Parameters</h2>
                <table>
                    <thead><tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">max_depth</code></td><td>int | None</td><td>None</td><td>Maximum tree depth (None = grow until pure)</td></tr>
                        <tr><td><code className="inline-code">what_lense</code></td><td>bool</td><td>False</td><td>Attach theoretical explanation to results</td></tr>
                        <tr><td><code className="inline-code">how_lense</code></td><td>bool</td><td>False</td><td>Attach step-by-step trace to results</td></tr>
                    </tbody>
                </table>

                <h2 id="returns">Returns: ModelResult</h2>
                <table>
                    <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">value</code></td><td>ndarray[int]</td><td>Predicted class labels</td></tr>
                        <tr><td><code className="inline-code">what_lense</code></td><td>str</td><td>Theoretical explanation of CART splitting</td></tr>
                        <tr><td><code className="inline-code">how_lense</code></td><td>str</td><td>Tree construction trace</td></tr>
                    </tbody>
                </table>
            </GuideLayout>
        </>
    );
}
