// Generic linalg function page template — used for operations with similar structure
import Head from 'next/head';
import GuideLayout from '../../../components/GuideLayout';
import CodeBlock from '../../../components/CodeBlock';

interface FnPageProps {
    name: string;
    title: string;
    tagline: string;
    what: string;
    mlUses: string[];
    complexity: string;
    codeExample: string;
    params: { name: string; type: string; desc: string }[];
    returns: string;
    prev: { label: string; href: string };
    next: { label: string; href: string };
}

function FnPage({ name, title, tagline, what, mlUses, complexity, codeExample, params, returns, prev, next }: FnPageProps) {
    return (
        <>
            <Head><title>{name} — ML-Lens math.linalg</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'math.linalg', href: '/guide/linalg' }, { label: name, href: `/guide/linalg/${name.replace('_', '-')}` }]}
                toc={[{ id: 'what', label: 'What is it?' }, { id: 'ml', label: 'Where in ML?' }, { id: 'usage', label: 'Usage' }, { id: 'params', label: 'Parameters' }]}
                prev={prev}
                next={next}
            >
                <h1>{title}</h1>
                <blockquote>{tagline}</blockquote>
                <h2 id="what">What is it?</h2>
                <p>{what}</p>
                <p><strong>Complexity:</strong> {complexity}</p>
                <h2 id="ml">Where used in ML?</h2>
                <ul>{mlUses.map(u => <li key={u}>{u}</li>)}</ul>
                <h2 id="usage">Usage</h2>
                <CodeBlock code={codeExample} />
                <h2 id="params">Parameters</h2>
                <table>
                    <thead><tr><th>Parameter</th><th>Type</th><th>Description</th></tr></thead>
                    <tbody>{params.map(p => <tr key={p.name}><td><code className="inline-code">{p.name}</code></td><td>{p.type}</td><td>{p.desc}</td></tr>)}</tbody>
                </table>
                <p><strong>Returns:</strong> {returns}</p>
            </GuideLayout>
        </>
    );
}

export default function SolvePage() {
    return <FnPage
        name="solve"
        title="solve"
        tagline="Solve the linear system Ax = b using Gaussian elimination with partial pivoting."
        what="Given a square matrix A and vector b, solve() finds vector x such that Ax = b. It uses LU decomposition / Gaussian elimination internally, with partial pivoting to avoid numerical instability from near-zero pivots."
        mlUses={['Computing OLS regression coefficients without explicitly inverting X^T X', 'Solving Kalman filter update equations', 'Finding optimal dual variables in SVMs', 'Any system of linear equations from physics or optimization']}
        complexity="O(n³) via Gaussian elimination"
        codeExample={`from mllense.math.linalg import solve

A = [[3, 1], [1, 2]]
b = [9, 8]

x = solve(A, b, what_lense=True, how_lense=True)
print(x)           # → [2.0, 3.0]
print(x.how_lense) # Step-by-step elimination trace`}
        params={[
            { name: 'a', type: 'MatrixLike', desc: 'Coefficient matrix A (must be square and non-singular)' },
            { name: 'b', type: 'VectorLike', desc: 'Right-hand side vector b' },
            { name: 'what_lense', type: 'bool', desc: 'Enable theoretical explanation' },
            { name: 'how_lense', type: 'bool', desc: 'Enable elimination step trace' },
        ]}
        returns="LinalgResult wrapping the solution vector x"
        prev={{ label: 'matmul', href: '/guide/linalg/matmul' }}
        next={{ label: 'add', href: '/guide/linalg/add' }}
    />;
}
