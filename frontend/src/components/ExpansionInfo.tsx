import styles from './ExpansionInfo.module.css';

interface ExpansionInfoProps {
  originalQuery: string;
  expandedQuery: string;
  abbreviations: string[];
}

export default function ExpansionInfo({ 
  originalQuery, 
  expandedQuery, 
  abbreviations 
}: ExpansionInfoProps) {
  if (originalQuery === expandedQuery) {
    return null;
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>Query Expansion</div>
      <div className={styles.text}>
        <span className={styles.original}>{originalQuery}</span>
        <span className={styles.arrow}> → </span>
        <span className={styles.expanded}>{expandedQuery}</span>
      </div>
      {abbreviations.length > 0 && (
        <div className={styles.tags}>
          {abbreviations.map((abbrev, idx) => (
            <span key={idx} className={styles.tag}>{abbrev}</span>
          ))}
        </div>
      )}
    </div>
  );
}

