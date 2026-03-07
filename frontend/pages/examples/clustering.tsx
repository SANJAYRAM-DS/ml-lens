import Head from 'next/head';
import GuideLayout from '../../components/GuideLayout';
import CodeBlock from '../../components/CodeBlock';

export default function ClusteringExample() {
    return (
        <>
            <Head><title>Clustering Example — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Examples', href: '/examples' }, { label: 'Clustering', href: '/examples/clustering' }]}
                toc={[]}
                prev={{ label: 'Back to Examples', href: '/examples' }}
            >
                <h1>Clustering with KMeans</h1>
                <p>This snippet initializes random centroids and assigns groups iteratively.</p>
                <CodeBlock code={`from mllense.models import KMeans\nimport numpy as np\n\nX = np.random.randn(300, 2)\nmodel = KMeans(n_clusters=4, how_lense=True)\nmodel.fit(X)\nprint("Cluster centers:", model.cluster_centers_)\nprint(model.how_lense)`} filename="kmeans_demo.py" />
            </GuideLayout>
        </>
    );
}
