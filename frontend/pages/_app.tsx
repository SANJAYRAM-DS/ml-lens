import type { AppProps } from 'next/app';
import Head from 'next/head';
import '../styles/globals.css';
import 'prismjs/themes/prism-tomorrow.css';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/logo.png" />
        <meta name="theme-color" content="#0d1117" />
      </Head>
      <Navbar />
      <Component {...pageProps} />
      <Footer />
    </>
  );
}
