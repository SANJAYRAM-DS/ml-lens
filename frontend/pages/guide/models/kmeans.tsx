import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

const toc = [{ id: 'what', label: 'What is it?' }, { id: 'ml', label: 'Where used in ML?' }, { id: 'how', label: 'How it works' }, { id: 'usage', label: 'Usage' }, { id: 'lenses', label: 'Lenses' }, { id: 'params', label: 'Parameters' }, { id: 'attrs', label: 'Attributes' }];

export default function KMeansPage() {
    return (
        <>
            <Head><title>KMeans — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'Models', href: '/guide/models' }, { label: 'KMeans', href: '/guide/models/kmeans' }]}
                toc={toc}
                prev={{ label: 'RandomForestRegressor', href: '/guide/models/random-forest-regressor' }}
                next={{ label: 'math.linalg Overview', href: '/guide/linalg' }}
            >
                <h1>KMeans</h1>
                <blockquote>Unsupervised clustering via iterative centroid assignment — partitions data into k groups minimizing intra-cluster Euclidean distance.</blockquote>

                <h2 id="what">What is it?</h2>
                <p>K-Means is the most widely used clustering algorithm. It partitions n samples into k clusters by iteratively alternating between two steps:</p>
                <ol>
                    <li><strong>Assignment:</strong> Each point is assigned to the nearest centroid (by Euclidean distance)</li>
                    <li><strong>Update:</strong> Each centroid is recomputed as the geometric mean of its assigned points</li>
                </ol>
                <p>Convergence happens when centroids stop moving. The objective minimized is the <strong>Within-Cluster Sum of Squares (WCSS)</strong>.</p>

                <h2 id="ml">Where Used in ML?</h2>
                <ul>
                    <li>Customer segmentation, document clustering, image compression</li>
                    <li>Feature engineering via cluster membership features</li>
                    <li>Initialization of Gaussian Mixture Models</li>
                    <li>Anomaly detection (high-distance points = outliers)</li>
                </ul>

                <h2 id="how">How it works internally</h2>
                <ol>
                    <li>Initialize k centroids using the Forgy method (random sample from data)</li>
                    <li>Compute Euclidean distance from every point to every centroid: O(k·n·d)</li>
                    <li>Assign each point to the nearest centroid</li>
                    <li>Recalculate each centroid as the mean of its assigned points</li>
                    <li>Repeat until centroids converge (np.allclose) or max_iter reached</li>
                </ol>
                <p><strong>Complexity:</strong> O(max_iter × k × n_samples × n_features)</p>

                <h2 id="usage">Basic Usage</h2>
                <CodeBlock code={`from mllense.models import KMeans
import numpy as np

X = np.random.randn(300, 2)

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

print(model.labels_)          # Cluster assignment for each point: [0, 2, 1, ...]
print(model.cluster_centers_) # Centroid coordinates for each cluster
labels = model.predict(X)     # Predict clusters for new data
labels = model.fit_predict(X) # Fit + predict in one call`} />

                <h2 id="lenses">Using the Lenses</h2>
                <CodeBlock code={`model = KMeans(n_clusters=3, how_lense=True, what_lense=True)
model.fit(X)
# After fit, access lens data:
print(model.what_lense)
# → "=== WHAT: k_means_clustering ===
#    K-Means splits samples into k clusters..."
print(model.how_lense)
# → "1. Running KMeans initialization (k=3)
#    2. Selecting 3 random indices as initial centroids
#    3. Iter 1: Computing Euclidean distance to assign points...
#    4. Iter 1: Recalculating centroids as geometric center...
#    ... Hiding repeating iteration logs ...
#    N. Centroids no longer moving. Converged after X iterations"`} />

                <h2 id="params">Parameters</h2>
                <table>
                    <thead><tr><th>Parameter</th><th>Type</th><th>Default</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">n_clusters</code></td><td>int</td><td>8</td><td>Number of clusters k</td></tr>
                        <tr><td><code className="inline-code">max_iter</code></td><td>int</td><td>300</td><td>Maximum number of iterations</td></tr>
                        <tr><td><code className="inline-code">random_state</code></td><td>int | None</td><td>None</td><td>Seed for reproducible centroid initialization</td></tr>
                        <tr><td><code className="inline-code">what_lense</code></td><td>bool</td><td>False</td><td>Enable theoretical explanations</td></tr>
                        <tr><td><code className="inline-code">how_lense</code></td><td>bool</td><td>False</td><td>Enable iteration trace</td></tr>
                    </tbody>
                </table>

                <h2 id="attrs">Fitted Attributes</h2>
                <table>
                    <thead><tr><th>Attribute</th><th>Type</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">cluster_centers_</code></td><td>ndarray (k, n_features)</td><td>Centroid coordinates after fitting</td></tr>
                        <tr><td><code className="inline-code">labels_</code></td><td>ndarray (n_samples,)</td><td>Cluster assignment for each training sample</td></tr>
                    </tbody>
                </table>
            </GuideLayout>
        </>
    );
}
