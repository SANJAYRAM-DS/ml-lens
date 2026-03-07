import Link from 'next/link';
import { useState, useEffect } from 'react';
import { Menu, X } from 'lucide-react';
import styles from './Navbar.module.css';

const navLinks = [
    { label: 'Home', href: '/' },
    { label: 'User Guide', href: '/guide' },
    { label: 'API', href: '/api-reference' },
    { label: 'Examples', href: '/examples' },
];

export default function Navbar() {
    const [open, setOpen] = useState(false);
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const handler = () => setScrolled(window.scrollY > 12);
        window.addEventListener('scroll', handler);
        return () => window.removeEventListener('scroll', handler);
    }, []);

    return (
        <nav className={`${styles.navbar} ${scrolled ? styles.scrolled : ''}`}>
            <div className={styles.inner}>
                {/* Logo */}
                <Link href="/" className={styles.logo}>
                    <img src="/logo.png" alt="ML-Lens" className={styles.logoImg} />
                    <div className={styles.logoText}>
                        <span className={styles.logoName}>ML-Lens</span>
                        <span className={styles.version}>v0.1.0</span>
                    </div>
                </Link>

                {/* Desktop Nav */}
                <div className={styles.links}>
                    {navLinks.map(l => (
                        <Link key={l.href} href={l.href} className={styles.navLink}>{l.label}</Link>
                    ))}
                    <Link href="/install" className={styles.installBtn}>
                        Install ›
                    </Link>
                </div>

                {/* Mobile Hamburger */}
                <button className={styles.hamburger} onClick={() => setOpen(!open)} aria-label="Menu">
                    {open ? <X size={20} /> : <Menu size={20} />}
                </button>
            </div>

            {/* Mobile Drawer */}
            {open && (
                <div className={styles.drawer}>
                    {navLinks.map(l => (
                        <Link key={l.href} href={l.href} className={styles.drawerLink} onClick={() => setOpen(false)}>{l.label}</Link>
                    ))}
                    <Link href="/install" className={styles.drawerInstall} onClick={() => setOpen(false)}>Install</Link>
                </div>
            )}
        </nav>
    );
}
