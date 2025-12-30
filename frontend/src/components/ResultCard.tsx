import { ProductMatch } from '@/types/api';
import styles from './ResultCard.module.css';

interface ResultCardProps {
  product: ProductMatch;
  index: number;
}

function getRankEmoji(rank: number): string {
  switch (rank) {
    case 1: return '🥇';
    case 2: return '🥈';
    case 3: return '🥉';
    default: return '🎯';
  }
}

export default function ResultCard({ product, index }: ResultCardProps) {
  const similarityPercent = (product.similarity_score * 100).toFixed(1);
  const isRanked = product.llm_rank !== null;
  
  return (
    <div 
      className={`${styles.card} ${isRanked ? styles.ranked : ''} ${product.llm_rank === 1 ? styles.topRanked : ''}`}
      style={{ animationDelay: `${index * 0.05}s` }}
    >
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <span className={styles.articleNumber}>{product.article_number}</span>
          {isRanked && (
            <span className={styles.rankBadge}>
              {getRankEmoji(product.llm_rank!)} #{product.llm_rank}
            </span>
          )}
        </div>
        <span className={styles.score}>{similarityPercent}%</span>
      </div>
      
      {/* Headline */}
      <p className={styles.headline}>{product.headline}</p>
      
      {/* Structured features - only show non-null values */}
      <div className={styles.features}>
        {product.category && <span className={styles.feature}>📦 {product.category}</span>}
        {product.size && <span className={styles.feature}>📏 {product.size}</span>}
        {product.materials && <span className={styles.feature}>🔩 {product.materials}</span>}
        {product.standard && <span className={styles.feature}>📋 {product.standard}</span>}
        {product.dimensions && <span className={styles.feature}>📐 {product.dimensions}</span>}
        {product.brand && <span className={styles.feature}>🏷️ {product.brand}</span>}
        {product.application && <span className={styles.feature}>🔧 {product.application}</span>}
      </div>
      
      {/* Long description (truncated) */}
      {product.long_description && (
        <p className={styles.longDescription}>
          {product.long_description.length > 200
            ? `${product.long_description.substring(0, 200)}...`
            : product.long_description
          }
        </p>
      )}
    </div>
  );
}
