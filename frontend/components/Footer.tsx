import Link from 'next/link';
import { Github, Package } from 'lucide-react';
import styles from './Footer.module.css';

export default function Footer() {
    return (
        <footer className={styles.footer}>
            <div className={styles.inner}>
                <div className={styles.brand}>
                    <img src="/logo.png" alt="ML-Lens" className={styles.logo} />
                    <p className={styles.tagline}>Built with ♥ for learners and researchers</p>
                    <div className={styles.socials}>
                        <a href="https://github.com/SANJAYRAM-DS/ml-lens" target="_blank" rel="noopener" className={styles.social}><Github size={16} /> GitHub</a>
                        <a href="https://pypi.org/project/mllense/" target="_blank" rel="noopener" className={styles.social}><Package size={16} /> PyPI</a>
                    </div>
                </div>
                <div className={styles.links}>
                    <div className={styles.col}>
                        <div className={styles.colTitle}>Docs</div>
                        <Link href="/guide">User Guide</Link>
                        <Link href="/guide/models">Models API</Link>
                        <Link href="/guide/linalg">math.linalg</Link>
                        <Link href="/guide/lenses">Lenses</Link>
                        <Link href="/guide/config">GlobalConfig</Link>
                    </div>
                    <div className={styles.col}>
                        <div className={styles.colTitle}>Reference</div>
                        <Link href="/api-reference">Full API</Link>
                        <Link href="/examples">Examples</Link>
                        <Link href="/install">Installation</Link>
                    </div>
                    <div className={styles.col}>
                        <div className={styles.colTitle}>Models</div>
                        <Link href="/guide/models/linear-regression">LinearRegression</Link>
                        <Link href="/guide/models/logistic-regression">LogisticRegression</Link>
                        <Link href="/guide/models/decision-tree-classifier">DecisionTree</Link>
                        <Link href="/guide/models/kmeans">KMeans</Link>
                    </div>
                </div>
            </div>
            <div className={styles.bottom}>
                <span>© 2024 ML-Lens · MIT License</span>
                <span className={styles.pip}>pip install mllense</span>
            </div>
        </footer>
    );
}
