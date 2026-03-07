import { useState, useEffect } from 'react';
import { Copy, Check } from 'lucide-react';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import styles from './CodeBlock.module.css';

interface CodeBlockProps {
    code: string;
    language?: string;
    filename?: string;
}

export default function CodeBlock({ code, language = 'python', filename }: CodeBlockProps) {
    const [copied, setCopied] = useState(false);

    useEffect(() => {
        Prism.highlightAll();
    }, [code, language]);

    const handleCopy = () => {
        navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className={styles.wrapper}>
            {filename && <div className={styles.filename}>{filename}</div>}
            <div className={styles.block}>
                <button className={styles.copyBtn} onClick={handleCopy} aria-label="Copy code">
                    {copied ? <><Check size={12} /> Copied!</> : <><Copy size={12} /> Copy</>}
                </button>
                <pre><code className={`language-${language}`}>{code}</code></pre>
            </div>
        </div>
    );
}
