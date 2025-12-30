'use client';

import { useState, useEffect } from 'react';
import { RFQFileEntry, RFQItem, ProductMatch } from '@/types/api';
import { searchForRFQItem } from '@/lib/api';
import styles from './RFQDetail.module.css';

interface RFQDetailProps {
  file: RFQFileEntry;
  onBack: () => void;
}

interface ItemWithMatches extends RFQItem {
  matches: ProductMatch[];
  isLoading: boolean;
  isExpanded: boolean;
  error?: string;
}

export default function RFQDetail({ file, onBack }: RFQDetailProps) {
  const [items, setItems] = useState<ItemWithMatches[]>([]);
  const [loadingAll, setLoadingAll] = useState(false);

  // Initialize items without matches
  useEffect(() => {
    setItems(
      file.items.map((item) => ({
        ...item,
        matches: [],
        isLoading: false,
        isExpanded: false,
      }))
    );
  }, [file]);

  const loadMatchesForItem = async (index: number) => {
    const item = items[index];
    if (item.matches.length > 0 || item.isLoading) return;

    setItems((prev) =>
      prev.map((it, i) => (i === index ? { ...it, isLoading: true } : it))
    );

    try {
      const result = await searchForRFQItem(item.search_query, 5);
      setItems((prev) =>
        prev.map((it, i) =>
          i === index
            ? { ...it, matches: result.matches, isLoading: false, isExpanded: true }
            : it
        )
      );
    } catch (err) {
      setItems((prev) =>
        prev.map((it, i) =>
          i === index
            ? { ...it, error: 'Failed to load matches', isLoading: false }
            : it
        )
      );
    }
  };

  const toggleExpand = (index: number) => {
    const item = items[index];
    
    if (!item.isExpanded && item.matches.length === 0) {
      loadMatchesForItem(index);
    } else {
      setItems((prev) =>
        prev.map((it, i) =>
          i === index ? { ...it, isExpanded: !it.isExpanded } : it
        )
      );
    }
  };

  const loadAllMatches = async () => {
    setLoadingAll(true);
    
    // Load matches for all items that don't have them yet
    const promises = items.map((item, index) => {
      if (item.matches.length === 0 && !item.isLoading) {
        return loadMatchesForItem(index);
      }
      return Promise.resolve();
    });

    await Promise.all(promises);
    
    // Expand all items
    setItems((prev) => prev.map((it) => ({ ...it, isExpanded: true })));
    setLoadingAll(false);
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <button className={styles.backBtn} onClick={onBack}>
          ← Back to files
        </button>
        <div className={styles.fileInfo}>
          <h1 className={styles.filename}>{file.filename}</h1>
          <span className={styles.meta}>
            {file.totalItems} items • {file.sourceType.toUpperCase()}
          </span>
        </div>
        <button 
          className={styles.loadAllBtn}
          onClick={loadAllMatches}
          disabled={loadingAll}
        >
          {loadingAll ? 'Loading...' : 'Load All Matches'}
        </button>
      </div>

      <div className={styles.itemsList}>
        {items.map((item, index) => (
          <div key={index} className={styles.itemCard}>
            <div
              className={styles.itemHeader}
              onClick={() => toggleExpand(index)}
            >
              <div className={styles.itemInfo}>
                <span className={styles.itemNumber}>#{index + 1}</span>
                <div className={styles.itemText}>
                  <p className={styles.rawText}>{item.raw_text}</p>
                  <p className={styles.searchQuery}>
                    🔍 {item.search_query}
                  </p>
                </div>
                {item.quantity && (
                  <span className={styles.quantity}>
                    {item.quantity} {item.unit || 'pcs'}
                  </span>
                )}
              </div>
              <div className={styles.expandIcon}>
                {item.isLoading ? (
                  <div className={styles.miniSpinner} />
                ) : (
                  <span className={item.isExpanded ? styles.expanded : ''}>
                    ▼
                  </span>
                )}
              </div>
            </div>

            {item.isExpanded && (
              <div className={styles.matchesContainer}>
                {item.error ? (
                  <p className={styles.error}>{item.error}</p>
                ) : item.matches.length === 0 ? (
                  <p className={styles.noMatches}>No matches found</p>
                ) : (
                  <div className={styles.matchesList}>
                    <div className={styles.matchesHeader}>Top Matches</div>
                    {item.matches.map((match, mIndex) => (
                      <div key={mIndex} className={styles.matchCard}>
                        <div className={styles.matchHeader}>
                          <span className={styles.articleNumber}>
                            {match.article_number}
                          </span>
                          <span className={styles.score}>
                            {(match.similarity_score * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className={styles.matchDescription}>
                          {match.combined_description.substring(0, 150)}
                          {match.combined_description.length > 150 ? '...' : ''}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

