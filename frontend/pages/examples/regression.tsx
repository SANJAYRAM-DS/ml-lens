import Head from 'next/head';
import GuideLayout from '../../components/GuideLayout';
import CodeBlock from '../../components/CodeBlock';

const code = `import numpy as np
from mllense.models import LinearRegression

# 1. Create simple dataset (e.g., house size vs price)
X = np.array([[1000], [1500], [2000], [2500]])
y = np.array([300000, 450000, 600000, 750000])

# 2. Linear Regression Model
model = LinearRegression(
    fit_intercept=True,
    what_lense=True,
    how_lense=True
)

# 3. Train the model
model.fit(X, y)

print("Learned Coefficient (price per sqft):", model.coef_[0])
print("Learned Intercept:", model.intercept_)

# 4. Prediction
X_test = np.array([[1800], [2200]])
result = model.predict(X_test)

print("\\nPredictions:", result.value)
print("\\n--- how_lense ---")
print(result.how_lense)`;

export default function RegressionExample() {
    return (
        <>
            <Head><title>Regression Example — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Examples', href: '/examples' }, { label: 'Regression', href: '/examples/regression' }]}
                toc={[]}
                prev={{ label: 'Back to Examples', href: '/examples' }}
            >
                <h1>Regression with LinearRegression</h1>
                <p>This example demonstrates the foundational linear regression algorithm using exact matrix inversion (the Normal Equations).</p>

                <CodeBlock code={code} filename="regression_demo.py" />

                <h2>Understanding the Trace</h2>
                <p>The <code className="inline-code">how_lense</code> trace will explain that the intercept was fitted by appending a column of 1s to the matrix, followed by solving <code className="inline-code">XᵀX = Xᵀy</code> natively via <code className="inline-code">math.linalg</code>.</p>
            </GuideLayout>
        </>
    );
}
