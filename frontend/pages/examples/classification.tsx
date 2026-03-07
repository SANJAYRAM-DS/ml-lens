import Head from 'next/head';
import GuideLayout from '../../components/GuideLayout';
import CodeBlock from '../../components/CodeBlock';

const code = `import numpy as np
from mllense.models import DecisionTreeClassifier

# 1. Create a toy dataset (e.g., student passing predicting based on hours studied & attendance)
X_train = np.array([
    [2, 0.5], [3, 0.8], [5, 0.9], [1, 0.2], [8, 1.0], [9, 0.8]
])
y_train = np.array([0, 0, 1, 0, 1, 1]) # 0=Fail, 1=Pass

# 2. Initialize classifier with lenses enabled
clf = DecisionTreeClassifier(
    max_depth=3,
    what_lense=True, 
    how_lense=True
)

# 3. Train
print("--- Training ---")
clf.fit(X_train, y_train)

# 4. Predict
X_test = np.array([[4, 0.85]])
result = clf.predict(X_test)

print("\\n--- Output ---")
print("Prediction:", result)
print("\\n--- what_lense ---")
print(result.what_lense)
print("\\n--- how_lense ---")
print(result.how_lense)`;

export default function ClassificationExample() {
    return (
        <>
            <Head><title>Classification Example — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Examples', href: '/examples' }, { label: 'Classification', href: '/examples/classification' }]}
                toc={[]}
                prev={{ label: 'Back to Examples', href: '/examples' }}
            >
                <h1>Classification with Decision Tree</h1>
                <p>This example demonstrates how to build a simple Decision Tree Classifier. It covers initialization, training, predicting, and importantly, observing the theoretical and operational traces.</p>

                <CodeBlock code={code} filename="classification_demo.py" />

                <h2>Expected Output</h2>
                <p>When you run this script, <code className="inline-code">how_lense</code> will step through how the tree splits the 2D feature space to maximize information gain, eventually assigning the test sample to the correct leaf node.</p>
            </GuideLayout>
        </>
    );
}
