import Link from 'next/link';
import { ReactNode } from 'react';
import styles from './GuideLayout.module.css';

interface SidebarItem {
    label: string;
    href: string;
    children?: { label: string; href: string }[];
}

const sidebar: SidebarItem[] = [
    { label: 'Overview', href: '/guide' },
    {
        label: 'Models',
        href: '/guide/models',
        children: [
            { label: 'LinearRegression', href: '/guide/models/linear-regression' },
            { label: 'LogisticRegression', href: '/guide/models/logistic-regression' },
            { label: 'DecisionTreeClassifier', href: '/guide/models/decision-tree-classifier' },
            { label: 'DecisionTreeRegressor', href: '/guide/models/decision-tree-regressor' },
            { label: 'RandomForestClassifier', href: '/guide/models/random-forest-classifier' },
            { label: 'RandomForestRegressor', href: '/guide/models/random-forest-regressor' },
            { label: 'KMeans', href: '/guide/models/kmeans' },
        ],
    },
    {
        label: 'math.linalg',
        href: '/guide/linalg',
        children: [
            { label: 'matmul', href: '/guide/linalg/matmul' },
            { label: 'solve', href: '/guide/linalg/solve' },
            { label: 'add', href: '/guide/linalg/add' },
            { label: 'subtract', href: '/guide/linalg/subtract' },
            { label: 'multiply', href: '/guide/linalg/multiply' },
            { label: 'divide', href: '/guide/linalg/divide' },
            { label: 'scalar_multiply', href: '/guide/linalg/scalar-multiply' },
            { label: 'scalar_add', href: '/guide/linalg/scalar-add' },
            { label: 'transpose', href: '/guide/linalg/transpose' },
            { label: 'reshape', href: '/guide/linalg/reshape' },
            { label: 'flatten', href: '/guide/linalg/flatten' },
            { label: 'vstack / hstack', href: '/guide/linalg/vstack' },
            { label: 'zeros / ones / eye', href: '/guide/linalg/creation' },
            { label: 'det', href: '/guide/linalg/det' },
            { label: 'inv', href: '/guide/linalg/inv' },
            { label: 'qr', href: '/guide/linalg/qr' },
            { label: 'svd', href: '/guide/linalg/svd' },
            { label: 'eig', href: '/guide/linalg/eig' },
            { label: 'matrix_trace', href: '/guide/linalg/matrix-trace' },
            { label: 'vector_norm', href: '/guide/linalg/vector-norm' },
            { label: 'frobenius_norm', href: '/guide/linalg/frobenius-norm' },
            { label: 'spectral_norm', href: '/guide/linalg/spectral-norm' },
            { label: 'condition_number', href: '/guide/linalg/condition-number' },
            { label: 'matrix_rank', href: '/guide/linalg/matrix-rank' },
            { label: 'dominant_eigen', href: '/guide/linalg/dominant-eigen' },
            { label: 'stability_report', href: '/guide/linalg/stability-report' },
        ],
    },
    { label: 'Lenses (what & how)', href: '/guide/lenses' },
    { label: 'GlobalConfig', href: '/guide/config' },
];

interface GuideLayoutProps {
    children: ReactNode;
    breadcrumbs?: { label: string; href: string }[];
    prev?: { label: string; href: string };
    next?: { label: string; href: string };
    toc?: { id: string; label: string }[];
}

export default function GuideLayout({ children, breadcrumbs, prev, next, toc }: GuideLayoutProps) {
    return (
        <div className={styles.layout}>
            {/* Sidebar */}
            <aside className={styles.sidebar}>
                <div className={styles.sidebarInner}>
                    {sidebar.map(item => (
                        <div key={item.href} className={styles.sidebarSection}>
                            <Link href={item.href} className={styles.sidebarParent}>{item.label}</Link>
                            {item.children && (
                                <div className={styles.sidebarChildren}>
                                    {item.children.map(child => (
                                        <Link key={child.href} href={child.href} className={styles.sidebarChild}>{child.label}</Link>
                                    ))}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </aside>

            {/* Main Content */}
            <main className={styles.content}>
                {breadcrumbs && (
                    <div className={styles.breadcrumbs}>
                        {breadcrumbs.map((b, i) => (
                            <span key={b.href}>
                                {i > 0 && <span className={styles.sep}> › </span>}
                                <Link href={b.href}>{b.label}</Link>
                            </span>
                        ))}
                    </div>
                )}
                <div className={styles.prose}>{children}</div>
                {(prev || next) && (
                    <div className={styles.prevNext}>
                        {prev ? <Link href={prev.href} className={styles.prevBtn}>← {prev.label}</Link> : <span />}
                        {next && <Link href={next.href} className={styles.nextBtn}>{next.label} →</Link>}
                    </div>
                )}
            </main>

            {/* Right TOC */}
            {toc && toc.length > 0 && (
                <aside className={styles.toc}>
                    <div className={styles.tocTitle}>On this page</div>
                    {toc.map(t => (
                        <a key={t.id} href={`#${t.id}`} className={styles.tocLink}>{t.label}</a>
                    ))}
                </aside>
            )}
        </div>
    );
}
