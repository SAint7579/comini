import { LLMRerankInfo, ProductMatch } from '@/types/api';
import styles from './LLMBestMatch.module.css';

interface LLMTopMatchesProps {
  rerank: LLMRerankInfo;
  matches: ProductMatch[];
}

function getConfidenceColor(confidence: string): string {
  switch (confidence.toLowerCase()) {
    case 'high':
      return 'var(--success)';
    case 'medium':
      return 'var(--accent)';
    case 'low':
      return 'var(--text-muted)';
    default:
      return 'var(--text-secondary)';
  }
}

function getRankEmoji(rank: number): string {
  switch (rank) {
    case 1: return '🥇';
    case 2: return '🥈';
    case 3: return '🥉';
    default: return '🎯';
  }
}

export default function LLMTopMatches({ rerank, matches }: LLMTopMatchesProps) {
  // Get the actual product for each ranked match
  const rankedProducts = rerank.top_matches.map((ranked, idx) => ({
    ...ranked,
    product: matches.find(m => m.article_number === ranked.article_number),
    rank: idx + 1,
  }));

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <span className={styles.icon}>🎯</span>
        <span className={styles.title}>LLM Top 3 Matches</span>
        <span 
          className={styles.confidence}
          style={{ color: getConfidenceColor(rerank.confidence) }}
        >
          {rerank.confidence} confidence
        </span>
      </div>
      
      <div className={styles.matchesList}>
        {rankedProducts.map(({ product, reasoning, rank }) => (
          product && (
            <div key={product.article_number} className={styles.matchCard}>
              <div className={styles.matchHeader}>
                <span className={styles.rankBadge}>
                  {getRankEmoji(rank)} #{rank}
                </span>
                <span className={styles.articleNumber}>{product.article_number}</span>
                <span className={styles.score}>
                  {(product.similarity_score * 100).toFixed(1)}%
                </span>
              </div>
              
              {/* Headline */}
              <p className={styles.headline}>{product.headline}</p>
              
              {/* Structured features - only show non-null values */}
              <div className={styles.features}>
                {product.category && <span className={styles.feature}>📦 {product.category}</span>}
                {product.size && <span className={styles.feature}>📏 {product.size}</span>}
                {product.materials && <span className={styles.feature}>🔩 {product.materials}</span>}
                {product.standard && <span className={styles.feature}>📋 {product.standard}</span>}
              </div>
              
              {/* Reasoning */}
              <p className={styles.reasoning}>{reasoning}</p>
            </div>
          )
        ))}
      </div>
    </div>
  );
}
