import Head from 'next/head';
import Link from 'next/link';
import CodeBlock from '../../components/CodeBlock';
import styles from './examples.module.css';

const examples = [
    {
        id: 'classification',
        title: 'Classification with Decision Tree',
        description: 'Train a DecisionTreeClassifier on the Iris dataset with full what_lense and how_lense explanations.',
        tags: ['Models', 'what_lense', 'how_lense'],
        preview: `from mllense.models import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=4, what_lense=True)
clf.fit(X_train, y_train)
result = clf.predict(X_test)`,
        href: '/examples/classification',
    },
    {
        id: 'regression',
        title: 'Regression with LinearRegression',
        description: 'Predict housing prices using LinearRegression with trace output showing matrix algebra.',
        tags: ['Models', 'how_lense'],
        preview: `from mllense.models import LinearRegression
model = LinearRegression(how_lense=True)
model.fit(X, y)
result = model.predict(X_test)
print(result.how_lense)`,
        href: '/examples/regression',
    },
    {
        id: 'clustering',
        title: 'Clustering with KMeans',
        description: 'Cluster customer data into 4 segments using KMeans with full iteration trace.',
        tags: ['Models', 'Unsupervised', 'how_lense'],
        preview: `from mllense.models import KMeans
model = KMeans(n_clusters=4, how_lense=True)
model.fit(X)
print(model.labels_)
print(model.how_lense)`,
        href: '/examples/clustering',
    },
    {
        id: 'linalg-walkthrough',
        title: 'Linear Algebra Walkthrough',
        description: 'Chain matmul → svd → eig with educational traces showing every computation step.',
        tags: ['linalg', 'what_lense', 'how_lense'],
        preview: `from mllense.math.linalg import matmul, svd, eig, transpose
C = matmul(A, B, how_lense=True)
U, S, Vt = svd(C.value, what_lense=True)
vals, vecs = eig(matmul(C.value, transpose(C.value)).value)`,
        href: '/examples/linalg-walkthrough',
    },
    {
        id: 'full-pipeline',
        title: 'Full ML Pipeline',
        description: 'End-to-end preprocessing + RandomForest + lenses — model the complete workflow.',
        tags: ['Models', 'linalg', 'Pipeline'],
        preview: `from mllense.math.linalg import matmul, transpose
from mllense.models import RandomForestClassifier
# Standardize manually: (X - mean) / std
model = RandomForestClassifier(n_estimators=100, what_lense=True)
model.fit(X_train, y)
result = model.predict(X_test)`,
        href: '/examples/full-pipeline',
    },
    {
        id: 'svd-pca',
        title: 'PCA via SVD',
        description: 'Implement principal component analysis from scratch using mllense SVD decomposition.',
        tags: ['linalg', 'svd', 'what_lense'],
        preview: `from mllense.math.linalg import svd, transpose, matmul
X_centered = X - X.mean(axis=0)
U, S, Vt = svd(X_centered, what_lense=True)
X_pca = matmul(X_centered, Vt[:2].T)`,
        href: '/examples/linalg-walkthrough',
    },
];

const tagColors: Record<string, string> = { Models: 'badge-purple', linalg: 'badge-blue', 'what_lense': 'badge-green', 'how_lense': 'badge-green', Unsupervised: 'badge-blue', Pipeline: 'badge-purple', svd: 'badge-blue' };

export default function ExamplesPage() {
    return (
        <>
            <Head>
                <title>Examples — ML-Lens</title>
                <meta name="description" content="Real-world examples showing mllense models, linear algebra, and tracing in action." />
            </Head>
            <div className={styles.page}>
                <div className={styles.header}>
                    <h1>Examples</h1>
                    <p className={styles.sub}>Real-world demonstrations of every major mllense feature — models, math, and lenses.</p>
                </div>
                <div className={styles.grid}>
                    {examples.map(ex => (
                        <Link href={ex.href} key={ex.id} className={styles.card}>
                            <div className={styles.tags}>{ex.tags.map(t => <span key={t} className={`badge ${tagColors[t] || 'badge-purple'}`}>{t}</span>)}</div>
                            <h2 className={styles.cardTitle}>{ex.title}</h2>
                            <p className={styles.cardDesc}>{ex.description}</p>
                            <pre className={styles.preview}><code>{ex.preview}</code></pre>
                            <span className={styles.viewLink}>View Example →</span>
                        </Link>
                    ))}
                </div>
            </div>
        </>
    );
}
