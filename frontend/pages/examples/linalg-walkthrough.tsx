import Head from 'next/head';
import GuideLayout from '../../components/GuideLayout';
import CodeBlock from '../../components/CodeBlock';

export default function LinalgWalkthroughExample() {
    return (
        <>
            <Head><title>Linear Algebra Walkthrough — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Examples', href: '/examples' }, { label: 'Linalg Walkthrough', href: '/examples/linalg-walkthrough' }]}
                toc={[]}
                prev={{ label: 'Back to Examples', href: '/examples' }}
            >
                <h1>Linear Algebra Walkthrough</h1>
                <p>Chain several math.linalg operations while retaining trace logic.</p>
                <CodeBlock code={`from mllense.math.linalg import matmul, svd, eig\n\nA = [[1, 2], [3, 4]]\nB = [[5, 6], [7, 8]]\n\nC = matmul(A, B, how_lense=True)\nprint("matmul trace:", C.how_lense)\n\nU, S, Vt = svd(C.value, what_lense=True)\nprint("svd theory:", U.what_lense)`} filename="math_chain.py" />
            </GuideLayout>
        </>
    );
}
