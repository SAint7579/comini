import { LLMRerankInfo, ProductMatch } from '@/types/api';
import styles from './LLMBestMatch.module.css';

interface LLMBestMatchProps {
  rerank: LLMRerankInfo;
  bestMatch?: ProductMatch;
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

export default function LLMBestMatch({ rerank, bestMatch }: LLMBestMatchProps) {
  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <span className={styles.icon}>🎯</span>
        <span className={styles.title}>LLM Best Match</span>
        <span 
          className={styles.confidence}
          style={{ color: getConfidenceColor(rerank.confidence) }}
        >
          {rerank.confidence} confidence
        </span>
      </div>
      
      {bestMatch && (
        <div className={styles.matchCard}>
          <div className={styles.matchHeader}>
            <span className={styles.articleNumber}>{bestMatch.article_number}</span>
            <span className={styles.score}>
              {(bestMatch.similarity_score * 100).toFixed(1)}% similarity
            </span>
          </div>
          <p className={styles.description}>
            {bestMatch.combined_description.substring(0, 200)}
            {bestMatch.combined_description.length > 200 ? '...' : ''}
          </p>
        </div>
      )}
      
      <div className={styles.reasoning}>
        <span className={styles.reasoningLabel}>Reasoning:</span>
        <span className={styles.reasoningText}>{rerank.reasoning}</span>
      </div>
    </div>
  );
}

