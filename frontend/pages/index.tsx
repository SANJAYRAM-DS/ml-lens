import Head from 'next/head';
import Link from 'next/link';
import { ArrowRight, ExternalLink, Sparkles, GitBranch, Cpu, Database, Settings, Package } from 'lucide-react';
import styles from './Home.module.css';
import CodeBlock from '../components/CodeBlock';

const heroCode = `from mllense.models import LinearRegression
from mllense.math.linalg import matmul, GlobalConfig

# Enable tracing globally
GlobalConfig.default_mode = "educational"

# Train a model
model = LinearRegression(what_lense=True, how_lense=True)
model.fit(X_train, y_train)

result = model.predict(X_test)
print(result.value)        # → [2.1, 3.8, 5.0]
print(result.what_lense)   # "Linear regression uses..."
print(result.how_lense)    # "Step 1: Computed X^T @ X..."`;

const matmulCode = `from mllense.math.linalg import matmul

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

result = matmul(A, B, what_lense=True, how_lense=True)
print(result)           # [[19, 22], [43, 50]]
print(result.how_lense) # "Step 1: Validate shapes (2x2)@(2x2)→(2x2)..."`;

const features = [
    {
        icon: <Sparkles size={24} />,
        title: 'what_lense',
        desc: 'Theoretical context for every operation — what it is, why it exists, and where it is used in real ML pipelines.',
        badge: 'Educational'
    },
    {
        icon: <Cpu size={24} />,
        title: 'how_lense',
        desc: 'Step-by-step operational trace of every computation: matrix shapes, loop iterations, pivot swaps, convergence.',
        badge: 'Tracing'
    },
    {
        icon: <Database size={24} />,
        title: 'math.linalg',
        desc: 'Full NumPy-compatible tensor engine: matmul, SVD, QR, eigenvalues, norms and more — all traceable.',
        badge: 'Core Math'
    },
    {
        icon: <GitBranch size={24} />,
        title: 'Models API',
        desc: 'LinearRegression, LogisticRegression, Decision Trees, Random Forests, KMeans — Scikit-Learn style interface.',
        badge: 'ML Models'
    },
    {
        icon: <Settings size={24} />,
        title: 'GlobalConfig',
        desc: 'Toggle tracing, switch backends (NumPy / Python / Numba) and change execution mode globally or per-call.',
        badge: 'Config'
    },
    {
        icon: <Package size={24} />,
        title: 'Pure Python',
        desc: 'Install in seconds: pip install mllense. Only NumPy required. Full Python 3.8+ support.',
        badge: 'PyPI'
    },
];

const models = [
    { name: 'LinearRegression', desc: 'OLS linear regression via normal equations', category: 'Supervised', href: '/guide/models/linear-regression' },
    { name: 'LogisticRegression', desc: 'Binary classification via gradient descent sigmoid', category: 'Supervised', href: '/guide/models/logistic-regression' },
    { name: 'DecisionTreeClassifier', desc: 'CART splitting on information gain', category: 'Supervised', href: '/guide/models/decision-tree-classifier' },
    { name: 'DecisionTreeRegressor', desc: 'CART splitting on MSE variance', category: 'Supervised', href: '/guide/models/decision-tree-regressor' },
    { name: 'RandomForestClassifier', desc: 'Bagged ensemble of decision trees (majority vote)', category: 'Ensemble', href: '/guide/models/random-forest-classifier' },
    { name: 'RandomForestRegressor', desc: 'Bagged ensemble of regression trees (mean)', category: 'Ensemble', href: '/guide/models/random-forest-regressor' },
    { name: 'KMeans', desc: 'K-Means clustering via Euclidean distance minimization', category: 'Unsupervised', href: '/guide/models/kmeans' },
];

const linalgOps = ['matmul', 'solve', 'add', 'subtract', 'multiply', 'divide', 'scalar_multiply', 'scalar_add', 'transpose', 'reshape', 'flatten', 'vstack', 'hstack', 'det', 'inv', 'qr', 'svd', 'eig', 'matrix_trace', 'dominant_eigen', 'vector_norm', 'frobenius_norm', 'spectral_norm', 'condition_number', 'matrix_rank', 'stability_report', 'full_diagnostic_report'];

export default function Home() {
    return (
        <>
            <Head>
                <title>ML-Lens — Educational Observability for Machine Learning</title>
                <meta name="description" content="Trace every matrix multiply. Understand every prediction. mllense is a production-grade, educational ML and linear algebra framework built on NumPy." />
            </Head>

            <section className={styles.hero}>
                <div className={styles.heroOrbs}>
                    <div className={styles.orb1} />
                    <div className={styles.orb2} />
                </div>
                <div className={styles.heroInner}>
                    <div className={styles.heroLeft}>
                        <div className={styles.heroEyebrow}>
                            <span className={styles.pip}>pip install mllense</span>
                        </div>
                        <h1 className={styles.heroTitle}>
                            Educational Observability<br />
                            for <span className="gradient-text">Machine Learning</span>
                        </h1>
                        <p className={styles.heroSub}>
                            Trace everything around Machine Learning. Understand every prediction.
                            Built on NumPy. Designed to teach.
                        </p>
                        <div className={styles.heroCtas}>
                            <Link href="/guide" className="btn btn-primary">Get Started <ArrowRight size={15} /></Link>
                            <a href="https://pypi.org/project/mllense/" target="_blank" rel="noreferrer" className="btn btn-outline">View on PyPI <ExternalLink size={13} /></a>
                        </div>
                    </div>
                    <div className={`${styles.heroRight} animate-float`}>
                        <CodeBlock filename="example.py" code={heroCode} />
                    </div>
                </div>
            </section>

            <section className={styles.section}>
                <div className="container">
                    <div className={styles.sectionHeader}>
                        <h2 className={styles.sectionTitle}>Why ML-Lens?</h2>
                        <p className={styles.sectionSub}>Not just another ML library. A teaching-first framework that exposes the math under the hood.</p>
                    </div>
                    <div className={styles.featureGrid}>
                        {features.map(f => (
                            <div key={f.title} className={`card ${styles.featureCard}`}>
                                <div className={styles.featureIcon}>{f.icon}</div>
                                <span className="badge badge-purple">{f.badge}</span>
                                <h3 className={styles.featureTitle}>{f.title}</h3>
                                <p className={styles.featureDesc}>{f.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            <section className={styles.demoSection}>
                <div className="container">
                    <div className={styles.sectionHeader}>
                        <h2 className={styles.sectionTitle}>See Inside Your Model</h2>
                        <p className={styles.sectionSub}>Both models and math operations return a <code className="inline-code">result</code> with trace data attached.</p>
                    </div>
                    <div className={styles.demoGrid}>
                        <div>
                            <h3 className={styles.demoLabel}>📐 math.linalg</h3>
                            <CodeBlock code={matmulCode} />
                        </div>
                        <div>
                            <h3 className={styles.demoLabel}>🤖 mllense.models</h3>
                            <CodeBlock code={heroCode} />
                        </div>
                    </div>
                </div>
            </section>

            <section className={styles.section}>
                <div className="container">
                    <div className={styles.sectionHeader}>
                        <h2 className={styles.sectionTitle}>Models at a Glance</h2>
                        <p className={styles.sectionSub}>All models follow the Scikit-Learn interface — <code className="inline-code">fit()</code>, <code className="inline-code">predict()</code>, <code className="inline-code">score()</code>.</p>
                    </div>
                    <div className={styles.modelsGrid}>
                        {models.map(m => (
                            <Link href={m.href} key={m.name} className={`card ${styles.modelCard}`}>
                                <span className={`badge ${m.category === 'Unsupervised' ? 'badge-blue' : m.category === 'Ensemble' ? 'badge-green' : 'badge-purple'}`}>{m.category}</span>
                                <div className={styles.modelName}>{m.name}</div>
                                <div className={styles.modelDesc}>{m.desc}</div>
                                <span className={styles.viewDocs}>View Docs →</span>
                            </Link>
                        ))}
                    </div>
                </div>
            </section>

            <section className={styles.linalgStrip}>
                <div className="container">
                    <h2 className={styles.stripTitle}>The Engine Under the Hood</h2>
                    <p className={styles.stripSub}>27+ operations — all with what_lense & how_lense tracing support</p>
                    <div className={styles.opsGrid}>
                        {linalgOps.map(op => (
                            <span key={op} className={styles.opChip}>{op}</span>
                        ))}
                    </div>
                    <div style={{ textAlign: 'center', marginTop: '28px' }}>
                        <Link href="/guide/linalg" className="btn btn-outline" style={{ display: 'inline-flex' }}>
                            Explore math.linalg →
                        </Link>
                    </div>
                </div>
            </section>

            <section className={styles.installSection}>
                <div className={styles.installBox}>
                    <div className={styles.installCmd}>
                        <span className={styles.dollar}>$</span>
                        <span className={styles.installText}>pip install mllense</span>
                    </div>
                    <p className={styles.installNote}>Available on PyPI · MIT License · Python 3.8+</p>
                    <div style={{ display: 'flex', gap: 12, justifyContent: 'center', marginTop: 20 }}>
                        <Link href="/install" className="btn btn-primary">Installation Guide</Link>
                        <Link href="/guide" className="btn btn-outline">Quick Start</Link>
                    </div>
                </div>
            </section>
        </>
    );
}
