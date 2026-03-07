import Head from 'next/head';
import Link from 'next/link';
import { useState } from 'react';
import { Search, ExternalLink } from 'lucide-react';
import styles from './api-reference.module.css';

const entries = [
    // Models
    { name: 'LinearRegression', module: 'mllense.models', category: 'Models', desc: 'OLS linear regression via normal equations. fit(X,y), predict(X), score(X,y).', href: '/guide/models/linear-regression' },
    { name: 'LogisticRegression', module: 'mllense.models', category: 'Models', desc: 'Binary classification via gradient descent + sigmoid. fit(X,y), predict(X), predict_proba(X).', href: '/guide/models/logistic-regression' },
    { name: 'DecisionTreeClassifier', module: 'mllense.models', category: 'Models', desc: 'CART tree classification on information gain / Gini impurity.', href: '/guide/models/decision-tree-classifier' },
    { name: 'DecisionTreeRegressor', module: 'mllense.models', category: 'Models', desc: 'CART tree regression minimizing MSE variance at splits.', href: '/guide/models/decision-tree-regressor' },
    { name: 'RandomForestClassifier', module: 'mllense.models', category: 'Models', desc: 'Ensemble of bootstrapped CART classifiers with majority voting.', href: '/guide/models/random-forest-classifier' },
    { name: 'RandomForestRegressor', module: 'mllense.models', category: 'Models', desc: 'Ensemble of bootstrapped CART regressors with mean aggregation.', href: '/guide/models/random-forest-regressor' },
    { name: 'KMeans', module: 'mllense.models', category: 'Models', desc: 'K-Means clustering via Euclidean distance and centroid iteration.', href: '/guide/models/kmeans' },
    // Linalg ops
    { name: 'matmul', module: 'mllense.math.linalg', category: 'linalg', desc: 'Matrix multiplication (dot product). Supports 1D, 2D. Auto-selects algorithm.', href: '/guide/linalg/matmul' },
    { name: 'solve', module: 'mllense.math.linalg', category: 'linalg', desc: 'Solve Ax=b linear system via Gaussian elimination with partial pivoting.', href: '/guide/linalg/solve' },
    { name: 'add', module: 'mllense.math.linalg', category: 'linalg', desc: 'Element-wise addition of two matrices.', href: '/guide/linalg/add' },
    { name: 'subtract', module: 'mllense.math.linalg', category: 'linalg', desc: 'Element-wise subtraction A - B.', href: '/guide/linalg/add' },
    { name: 'multiply', module: 'mllense.math.linalg', category: 'linalg', desc: 'Hadamard (element-wise) product of two matrices.', href: '/guide/linalg/add' },
    { name: 'divide', module: 'mllense.math.linalg', category: 'linalg', desc: 'Element-wise division A / B.', href: '/guide/linalg/add' },
    { name: 'scalar_multiply', module: 'mllense.math.linalg', category: 'linalg', desc: 'Multiply every element by a scalar constant.', href: '/guide/linalg/add' },
    { name: 'scalar_add', module: 'mllense.math.linalg', category: 'linalg', desc: 'Add a scalar constant to every element.', href: '/guide/linalg/add' },
    { name: 'transpose', module: 'mllense.math.linalg', category: 'linalg', desc: 'Flip matrix along diagonal (rows become columns).', href: '/guide/linalg/transpose' },
    { name: 'reshape', module: 'mllense.math.linalg', category: 'linalg', desc: 'Rearrange elements into a new matrix shape.', href: '/guide/linalg/transpose' },
    { name: 'flatten', module: 'mllense.math.linalg', category: 'linalg', desc: 'Collapse matrix to a 1D vector (row-major).', href: '/guide/linalg/transpose' },
    { name: 'vstack', module: 'mllense.math.linalg', category: 'linalg', desc: 'Stack matrices vertically (row axis).', href: '/guide/linalg/transpose' },
    { name: 'hstack', module: 'mllense.math.linalg', category: 'linalg', desc: 'Stack matrices horizontally (column axis).', href: '/guide/linalg/transpose' },
    { name: 'zeros', module: 'mllense.math.linalg', category: 'linalg', desc: 'Create matrix filled with 0.0.', href: '/guide/linalg/transpose' },
    { name: 'ones', module: 'mllense.math.linalg', category: 'linalg', desc: 'Create matrix filled with 1.0.', href: '/guide/linalg/transpose' },
    { name: 'eye', module: 'mllense.math.linalg', category: 'linalg', desc: 'Create identity matrix of size n×n.', href: '/guide/linalg/transpose' },
    { name: 'rand', module: 'mllense.math.linalg', category: 'linalg', desc: 'Create matrix with uniform random values [0, 1).', href: '/guide/linalg/transpose' },
    { name: 'det', module: 'mllense.math.linalg', category: 'linalg', desc: 'Compute determinant of a square matrix.', href: '/guide/linalg/det' },
    { name: 'inv', module: 'mllense.math.linalg', category: 'linalg', desc: 'Compute matrix inverse A⁻¹ such that A @ A⁻¹ = I.', href: '/guide/linalg/det' },
    { name: 'matrix_trace', module: 'mllense.math.linalg', category: 'linalg', desc: 'Sum of diagonal elements of a square matrix.', href: '/guide/linalg/det' },
    { name: 'qr', module: 'mllense.math.linalg', category: 'linalg', desc: 'QR factorization: A = QR. Q is orthogonal, R is upper triangular.', href: '/guide/linalg/det' },
    { name: 'svd', module: 'mllense.math.linalg', category: 'linalg', desc: 'Singular Value Decomposition: A = U Σ Vᵀ.', href: '/guide/linalg/svd' },
    { name: 'eig', module: 'mllense.math.linalg', category: 'linalg', desc: 'Eigendecomposition: find eigenvalues and eigenvectors of a square matrix.', href: '/guide/linalg/det' },
    { name: 'dominant_eigen', module: 'mllense.math.linalg', category: 'linalg', desc: 'Power iteration for the largest eigenvalue/eigenvector pair.', href: '/guide/linalg/det' },
    { name: 'vector_norm', module: 'mllense.math.linalg', category: 'linalg', desc: 'P-norm of a 1D vector (L1, L2, or L-infinity).', href: '/guide/linalg/vector-norm' },
    { name: 'frobenius_norm', module: 'mllense.math.linalg', category: 'linalg', desc: 'Frobenius norm — square root of sum of squared elements.', href: '/guide/linalg/vector-norm' },
    { name: 'spectral_norm', module: 'mllense.math.linalg', category: 'linalg', desc: 'Spectral 2-norm — largest singular value of matrix.', href: '/guide/linalg/vector-norm' },
    { name: 'condition_number', module: 'mllense.math.linalg', category: 'linalg', desc: 'σ_max / σ_min — measures numerical stability of matrix.', href: '/guide/linalg/vector-norm' },
    { name: 'matrix_rank', module: 'mllense.math.linalg', category: 'linalg', desc: 'Number of linearly independent rows/columns via SVD threshold.', href: '/guide/linalg/vector-norm' },
    { name: 'stability_report', module: 'mllense.math.linalg', category: 'linalg', desc: 'Perturbation analysis report for numerical stability assessment.', href: '/guide/linalg/vector-norm' },
    { name: 'full_diagnostic_report', module: 'mllense.math.linalg', category: 'linalg', desc: 'Combined report: rank, condition number, and stability analysis.', href: '/guide/linalg/vector-norm' },
    { name: 'GlobalConfig', module: 'mllense.math.linalg', category: 'Config', desc: 'Singleton configuration for backend, mode, and trace settings.', href: '/guide/config' },
    { name: 'get_config', module: 'mllense.math.linalg', category: 'Config', desc: 'Helper to get the singleton GlobalConfig instance.', href: '/guide/config' },
];

const categories = ['All', 'Models', 'linalg', 'Config'];
const categoryColors: Record<string, string> = { Models: 'badge-purple', linalg: 'badge-blue', Config: 'badge-green' };

export default function APIReference() {
    const [query, setQuery] = useState('');
    const [cat, setCat] = useState('All');

    const filtered = entries.filter(e =>
        (cat === 'All' || e.category === cat) &&
        (e.name.toLowerCase().includes(query.toLowerCase()) || e.desc.toLowerCase().includes(query.toLowerCase()))
    );

    return (
        <>
            <Head>
                <title>API Reference — ML-Lens</title>
                <meta name="description" content="Complete, searchable API reference for all mllense models and math.linalg operations." />
            </Head>
            <div className={styles.page}>
                <div className={styles.header}>
                    <h1 className={styles.title}>API Reference</h1>
                    <p className={styles.sub}>Complete reference for all {entries.length} items in mllense — models, math operations, and configuration.</p>
                    <div className={styles.controls}>
                        <div className={styles.searchWrap}>
                            <Search size={16} className={styles.searchIcon} />
                            <input className={styles.search} placeholder="Search function or model..." value={query} onChange={e => setQuery(e.target.value)} />
                        </div>
                        <div className={styles.filters}>
                            {categories.map(c => (
                                <button key={c} onClick={() => setCat(c)} className={`${styles.filter} ${cat === c ? styles.active : ''}`}>{c}</button>
                            ))}
                        </div>
                    </div>
                </div>

                <div className={styles.tableWrap}>
                    <table className={styles.table}>
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Module</th>
                                <th>Category</th>
                                <th>Description</th>
                                <th></th>
                            </tr>
                        </thead>
                        <tbody>
                            {filtered.map(e => (
                                <tr key={e.name} className={styles.row}>
                                    <td><code className="inline-code">{e.name}</code></td>
                                    <td><span className={styles.modName}>{e.module}</span></td>
                                    <td><span className={`badge ${categoryColors[e.category]}`}>{e.category}</span></td>
                                    <td className={styles.desc}>{e.desc}</td>
                                    <td><Link href={e.href} className={styles.docLink}>Docs <ExternalLink size={12} /></Link></td>
                                </tr>
                            ))}
                            {filtered.length === 0 && (
                                <tr><td colSpan={5} className={styles.empty}>No results for "{query}"</td></tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </>
    );
}
