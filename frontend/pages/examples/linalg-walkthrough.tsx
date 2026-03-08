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
                <CodeBlock code={`from mllense.math.linalg import matmul, svd, eig, transpose\n\nA = [[1, 2], [3, 4]]\nB = [[5, 6], [7, 8]]\n\n# matmul returns a LinalgResult — use .value for the raw matrix\nC = matmul(A, B, how_lense=True)\nprint("matmul trace:", C.how_lense)\n\n# svd() accepts a plain matrix/list and returns (U, S, Vt) as plain arrays\nU, S, Vt = svd(C.value)\nprint("U (left singular vectors):", U)\nprint("S (singular values):", S)\n\n# eig() also returns (eigenvalues, eigenvectors) as plain arrays\nCCT = matmul(C.value, transpose(C.value))\nvals, vecs = eig(CCT.value)\nprint("Eigenvalues:", vals)`} filename="math_chain.py" />
            </GuideLayout>
        </>
    );
}
