import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

const funcs = [
    { name: 'add', code: 'from mllense.math.linalg import add\nC = add(A, B, what_lense=True)\nprint(C)  # Element-wise A + B', desc: 'Element-wise addition of two matrices (C[i,j] = A[i,j] + B[i,j]).' },
    { name: 'subtract', code: 'from mllense.math.linalg import subtract\nC = subtract(A, B)\nprint(C)', desc: 'Element-wise subtraction (C[i,j] = A[i,j] - B[i,j]).' },
    { name: 'multiply', code: 'from mllense.math.linalg import multiply\nC = multiply(A, B)  # Hadamard product\nprint(C)', desc: 'Hadamard (element-wise) product. NOT matrix multiplication.' },
    { name: 'divide', code: 'from mllense.math.linalg import divide\nC = divide(A, B)  # Element-wise A / B\nprint(C)', desc: 'Element-wise division (C[i,j] = A[i,j] / B[i,j]).' },
    { name: 'scalar_multiply', code: 'from mllense.math.linalg import scalar_multiply\nC = scalar_multiply(A, 3.0)\nprint(C)  # Every element × 3.0', desc: 'Multiply every element of a matrix by a scalar constant.' },
    { name: 'scalar_add', code: 'from mllense.math.linalg import scalar_add\nC = scalar_add(A, 5.0)\nprint(C)  # Every element + 5.0', desc: 'Add a scalar constant to every element of a matrix.' },
];

export default function OpsPage() {
    return (
        <>
            <Head><title>Element-wise Operations — ML-Lens math.linalg</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'math.linalg', href: '/guide/linalg' }, { label: 'Element-wise Ops', href: '/guide/linalg/add' }]}
                toc={funcs.map(f => ({ id: f.name, label: f.name }))}
                prev={{ label: 'solve', href: '/guide/linalg/solve' }}
                next={{ label: 'transpose', href: '/guide/linalg/transpose' }}
            >
                <h1>Element-wise Operations</h1>
                <p>These operations work element-by-element on matrices of the same shape. All support the <code className="inline-code">what_lense</code> and <code className="inline-code">how_lense</code> parameters and return a <code className="inline-code">LinalgResult</code>.</p>
                <p>All require matrices A and B to have the same shape.</p>
                {funcs.map(f => (
                    <div key={f.name}>
                        <h2 id={f.name}>{f.name}</h2>
                        <p>{f.desc}</p>
                        <CodeBlock code={f.code} />
                    </div>
                ))}
            </GuideLayout>
        </>
    );
}
