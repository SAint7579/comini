import { ProductMatch } from '@/types/api';
import ResultCard from './ResultCard';
import styles from './ResultsList.module.css';

interface ResultsListProps {
  results: ProductMatch[];
}

export default function ResultsList({ results }: ResultsListProps) {
  if (results.length === 0) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>🔍</div>
        <p>No products found matching your query</p>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <span className={styles.count}>{results.length} results found</span>
      </div>
      <div className={styles.list}>
        {results.map((product, index) => (
          <ResultCard 
            key={product.article_number} 
            product={product} 
            index={index}
          />
        ))}
      </div>
    </div>
  );
}

