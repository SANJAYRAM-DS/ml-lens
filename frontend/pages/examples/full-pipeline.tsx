import Head from 'next/head';
import GuideLayout from '../../components/GuideLayout';
import CodeBlock from '../../components/CodeBlock';

export default function FullPipelineExample() {
    return (
        <>
            <Head><title>Full ML Pipeline Example — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Examples', href: '/examples' }, { label: 'Full Pipeline', href: '/examples/full-pipeline' }]}
                toc={[]}
                prev={{ label: 'Back to Examples', href: '/examples' }}
            >
                <h1>Full ML Pipeline</h1>
                <p>Combine preprocessing via math.linalg and classification via models.</p>
                <CodeBlock code={`from mllense.math.linalg import matmul, transpose\nfrom mllense.models import RandomForestClassifier\nimport numpy as np\n\n# Standardize X manually (mllense has no built-in normalize)\nX_train = np.array(X_raw)\nX_mean = X_train.mean(axis=0)\nX_std = X_train.std(axis=0)\nX_norm = (X_train - X_mean) / X_std\n\nmodel = RandomForestClassifier(n_estimators=100, what_lense=True)\nmodel.fit(X_norm, y)\nresult = model.predict(X_test)\nprint(result.what_lense)`} filename="pipeline.py" />
            </GuideLayout>
        </>
    );
}
