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
                <CodeBlock code={`from mllense.math.linalg import matmul, transpose\nfrom mllense.models import RandomForestClassifier\n\n# Standardize data manually using linalg (X_centered / std)\n# ...\n\nmodel = RandomForestClassifier(n_estimators=50, what_lense=True)\nmodel.fit(X_train, y_train)\nresult = model.predict(X_test)\nprint(result.what_lense)`} filename="pipeline.py" />
            </GuideLayout>
        </>
    );
}
