import Head from 'next/head';
import Link from 'next/link';
import CodeBlock from '../components/CodeBlock';
import styles from './install.module.css';

export default function InstallPage() {
    return (
        <>
            <Head>
                <title>Installation — ML-Lens</title>
                <meta name="description" content="Install mllense via pip. Python 3.8+ and NumPy required." />
            </Head>
            <div className={styles.page}>
                <div className={styles.hero}>
                    <h1 className={styles.title}>Installation</h1>
                    <p className={styles.sub}>mllense is available on PyPI. One command gets you everything.</p>
                </div>

                <div className={styles.main}>
                    <section className={styles.section}>
                        <h2>Requirements</h2>
                        <div className={styles.reqGrid}>
                            {[['Python', '3.8+', 'Core runtime'], ['NumPy', '≥1.20', 'Linear algebra backend'], ['pip', 'any', 'Package manager']].map(([n, v, d]) => (
                                <div key={n} className={styles.reqCard}>
                                    <span className={styles.reqName}>{n}</span>
                                    <span className="badge badge-purple">{v}</span>
                                    <span className={styles.reqDesc}>{d}</span>
                                </div>
                            ))}
                        </div>
                    </section>

                    <section className={styles.section}>
                        <h2>Install via PyPI</h2>
                        <CodeBlock code="pip install mllense" />
                        <p className={styles.note}>Or with a specific version: <code className="inline-code">pip install mllense==0.1.0</code></p>
                    </section>

                    <section className={styles.section}>
                        <h2>Verify Installation</h2>
                        <CodeBlock code={`python -c "import mllense; print(mllense.__version__)"
# → 0.1.0`} />
                    </section>

                    <section className={styles.section}>
                        <h2>Quick Start</h2>
                        <CodeBlock filename="quickstart.py" code={`import numpy as np
from mllense.models import LinearRegression
from mllense.math.linalg import matmul

# --- 1. Try a model ---
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([2.0, 3.0, 4.0])

model = LinearRegression(what_lense=True, how_lense=True)
model.fit(X, y)
result = model.predict(X)
print("Predictions:", result)
print()
print("what_lense:", result.what_lense)
print()
print("how_lense:", result.how_lense)

# --- 2. Try a linalg operation ---
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = matmul(A, B, what_lense=True)
print("C =", C)
print("Explanation:", C.what_lense)`} />
                    </section>

                    <section className={styles.section}>
                        <h2>Enable Global Tracing</h2>
                        <CodeBlock code={`from mllense.math.linalg import GlobalConfig
GlobalConfig.default_mode = "educational"
GlobalConfig.trace_enabled = True  # Enables how_lense on all linalg ops`} />
                    </section>

                    <section className={styles.section}>
                        <h2>Optional: Development Install</h2>
                        <p>If you want to contribute or explore the source:</p>
                        <CodeBlock code={`git clone https://github.com/SANJAYRAM-DS/ml-lens.git
cd ml-lens
pip install -e ".[dev]"
pytest  # Run test suite`} />
                    </section>

                    <div className={styles.nextSteps}>
                        <h2>What's Next?</h2>
                        <div className={styles.links}>
                            <Link href="/guide" className="btn btn-primary">→ User Guide</Link>
                            <Link href="/examples" className="btn btn-outline">Browse Examples</Link>
                            <Link href="/api-reference" className="btn btn-outline">API Reference</Link>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
}
