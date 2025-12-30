import { ProductMatch } from '@/types/api';
import styles from './ResultCard.module.css';

interface ResultCardProps {
  product: ProductMatch;
  index: number;
}

export default function ResultCard({ product, index }: ResultCardProps) {
  const similarityPercent = (product.similarity_score * 100).toFixed(1);
  
  return (
    <div 
      className={styles.card}
      style={{ animationDelay: `${index * 0.05}s` }}
    >
      <div className={styles.header}>
        <span className={styles.articleNumber}>{product.article_number}</span>
        <span className={styles.score}>{similarityPercent}%</span>
      </div>
      
      <p className={styles.description}>
        {product.combined_description.length > 200 
          ? `${product.combined_description.substring(0, 200)}...`
          : product.combined_description
        }
      </p>
      
      {product.long_description && (
        <p className={styles.longDescription}>
          {product.long_description.length > 300
            ? `${product.long_description.substring(0, 300)}...`
            : product.long_description
          }
        </p>
      )}
    </div>
  );
}

