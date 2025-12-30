'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import styles from './Navigation.module.css';

export default function Navigation() {
  const pathname = usePathname();

  return (
    <nav className={styles.nav}>
      <div className={styles.container}>
        <Link href="/" className={styles.logo}>
          Comini
        </Link>
        <div className={styles.links}>
          <Link 
            href="/" 
            className={`${styles.link} ${pathname === '/' ? styles.active : ''}`}
          >
            Product Search
          </Link>
          <Link 
            href="/rfq" 
            className={`${styles.link} ${pathname === '/rfq' ? styles.active : ''}`}
          >
            RFQ Upload
          </Link>
        </div>
      </div>
    </nav>
  );
}

