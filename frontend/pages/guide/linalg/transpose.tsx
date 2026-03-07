import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

export default function TransposePage() {
    return (
        <>
            <Head><title>transpose, reshape, flatten — ML-Lens math.linalg</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'math.linalg', href: '/guide/linalg' }, { label: 'Shape Ops', href: '/guide/linalg/transpose' }]}
                toc={[{ id: 'transpose', label: 'transpose' }, { id: 'reshape', label: 'reshape' }, { id: 'flatten', label: 'flatten' }, { id: 'vstack', label: 'vstack / hstack' }, { id: 'creation', label: 'zeros / ones / eye / rand' }]}
                prev={{ label: 'Element-wise Ops', href: '/guide/linalg/add' }}
                next={{ label: 'det', href: '/guide/linalg/det' }}
            >
                <h1>Shape Operations</h1>
                <p>Functions for transforming the structure and layout of matrices without changing their underlying data.</p>

                <h2 id="transpose">transpose</h2>
                <p>Flips the matrix along its diagonal — rows become columns.</p>
                <CodeBlock code={`from mllense.math.linalg import transpose
A = [[1, 2, 3], [4, 5, 6]]   # shape (2, 3)
At = transpose(A)              # shape (3, 2)
# [[1, 4], [2, 5], [3, 6]]`} />

                <h2 id="reshape">reshape</h2>
                <p>Rearranges the elements of a matrix into a new shape (total elements must stay constant).</p>
                <CodeBlock code={`from mllense.math.linalg import reshape
A = [[1, 2], [3, 4], [5, 6]]  # shape (3, 2)
B = reshape(A, (2, 3))         # shape (2, 3)
# [[1, 2, 3], [4, 5, 6]]`} />

                <h2 id="flatten">flatten</h2>
                <p>Collapses an m×n matrix into a flat 1D vector of length m*n (row-major order).</p>
                <CodeBlock code={`from mllense.math.linalg import flatten
A = [[1, 2], [3, 4]]
v = flatten(A)
# [1, 2, 3, 4]`} />

                <h2 id="vstack">vstack / hstack</h2>
                <p>Stack matrices vertically (rows stacked) or horizontally (columns stacked).</p>
                <CodeBlock code={`from mllense.math.linalg import vstack, hstack
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
V = vstack((A, B))  # shape (4, 2)
H = hstack((A, B))  # shape (2, 4)`} />

                <h2 id="creation">zeros / ones / eye / rand</h2>
                <p>Matrix creation utilities.</p>
                <CodeBlock code={`from mllense.math.linalg import zeros, ones, eye, rand
Z = zeros((3, 4))   # 3x4 matrix of 0.0
O = ones((2, 2))    # 2x2 matrix of 1.0
I = eye(4)          # 4x4 identity matrix
R = rand((5, 5))    # 5x5 uniform random matrix [0, 1)`} />
            </GuideLayout>
        </>
    );
}
