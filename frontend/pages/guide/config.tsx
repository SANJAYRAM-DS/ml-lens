import Head from 'next/head';
import GuideLayout from '../../components/GuideLayout';
import CodeBlock from '../../components/CodeBlock';

export default function ConfigPage() {
    return (
        <>
            <Head><title>GlobalConfig — ML-Lens</title></Head>
            <GuideLayout
                breadcrumbs={[{ label: 'Guide', href: '/guide' }, { label: 'GlobalConfig', href: '/guide/config' }]}
                toc={[{ id: 'overview', label: 'Overview' }, { id: 'backend', label: 'default_backend' }, { id: 'mode', label: 'default_mode' }, { id: 'trace', label: 'trace_enabled' }, { id: 'auto', label: 'auto_algorithm_selection' }, { id: 'reset', label: 'reset()' }]}
                prev={{ label: 'Lenses', href: '/guide/lenses' }}
                next={{ label: 'API Reference', href: '/api-reference' }}
            >
                <h1>GlobalConfig</h1>
                <blockquote>Thread-safe singleton configuration for the entire mllense math engine.</blockquote>

                <h2 id="overview">Overview</h2>
                <p><code className="inline-code">GlobalConfig</code> is a thread-safe singleton that controls the default behavior of all <code className="inline-code">mllense.math.linalg</code> operations. Changes persist for the lifetime of the process. All per-call parameters override these defaults.</p>
                <CodeBlock code={`from mllense.math.linalg import GlobalConfig, get_config

# Access the singleton
cfg = GlobalConfig()        # Singleton instance
cfg = get_config()          # Equivalent helper function

# All settings visible at once
print(cfg)
# → GlobalConfig(default_backend='numpy', default_mode='fast',
#                trace_enabled=False, auto_algorithm_selection=True)`} />

                <h2 id="backend">default_backend</h2>
                <p>Controls which backend handles numeric computation:</p>
                <table>
                    <thead><tr><th>Value</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">"numpy"</code> (default)</td><td>Uses np.matmul, np.linalg — fast BLAS-backed</td></tr>
                        <tr><td><code className="inline-code">"python"</code></td><td>Pure Python list loops — slow but transparent/traceable</td></tr>
                        <tr><td><code className="inline-code">"numba"</code></td><td>JIT-compiled — require numba install; fast after warmup</td></tr>
                    </tbody>
                </table>
                <CodeBlock code={`GlobalConfig().default_backend = "python"  # Transparent loops
GlobalConfig().default_backend = "numpy"   # Production speed`} />

                <h2 id="mode">default_mode</h2>
                <table>
                    <thead><tr><th>Value</th><th>Description</th></tr></thead>
                    <tbody>
                        <tr><td><code className="inline-code">"fast"</code> (default)</td><td>Optimized: auto-selects best algorithm for size</td></tr>
                        <tr><td><code className="inline-code">"educational"</code></td><td>Forces naive algorithms for clarity, enables tracing</td></tr>
                        <tr><td><code className="inline-code">"debug"</code></td><td>All checks, verbose output, assertion-heavy</td></tr>
                    </tbody>
                </table>
                <CodeBlock code={`GlobalConfig().default_mode = "educational"  # Teaching mode`} />

                <h2 id="trace">trace_enabled</h2>
                <p>When True, enables step-by-step trace recording globally (equivalent to passing <code className="inline-code">how_lense=True</code> to every call).</p>
                <CodeBlock code={`GlobalConfig().trace_enabled = True   # Record traces everywhere
GlobalConfig().trace_enabled = False  # Disable (default)`} />

                <h2 id="auto">auto_algorithm_selection</h2>
                <p>When True (default), the algorithm registry automatically picks the optimal implementation based on matrix size and current backend. Disable to always use the explicitly specified algorithm.</p>
                <CodeBlock code={`GlobalConfig().auto_algorithm_selection = False  # Disable auto
# Now must specify algorithm manually per-call:`} />

                <h2 id="reset">reset()</h2>
                <p>Resets all settings to factory defaults. Useful in tests or notebook sessions.</p>
                <CodeBlock code={`cfg = get_config()
cfg.reset()
print(cfg)
# → GlobalConfig(default_backend='numpy', default_mode='fast',
#                trace_enabled=False, auto_algorithm_selection=True)`} />
            </GuideLayout>
        </>
    );
}
