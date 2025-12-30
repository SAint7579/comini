'use client';

import { useState, useEffect } from 'react';
import { RFQFileEntry, RFQItem, ProductMatch, LLMRerankInfo } from '@/types/api';
import { searchForRFQItem } from '@/lib/api';
import styles from './RFQDetail.module.css';

interface RFQDetailProps {
  file: RFQFileEntry;
  onBack: () => void;
}

interface ItemWithMatches extends RFQItem {
  matches: ProductMatch[];
  llmRerank: LLMRerankInfo | null;
  isLoading: boolean;
  isExpanded: boolean;
  error?: string;
}

function getRankEmoji(rank: number): string {
  switch (rank) {
    case 1: return '🥇';
    case 2: return '🥈';
    case 3: return '🥉';
    default: return '🎯';
  }
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
        llmRerank: null,
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
      const result = await searchForRFQItem(item.search_query, 20);
      setItems((prev) =>
        prev.map((it, i) =>
          i === index
            ? { 
                ...it, 
                matches: result.matches, 
                llmRerank: result.llm_rerank,
                isLoading: false, 
                isExpanded: true 
              }
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

  // Get top 3 LLM ranked matches
  const getTopMatches = (item: ItemWithMatches): { match: ProductMatch; rank: number; reasoning: string }[] => {
    if (!item.llmRerank) return [];
    
    return item.llmRerank.top_matches.map((ranked, idx) => {
      const match = item.matches.find(m => m.article_number === ranked.article_number);
      return match ? { match, rank: idx + 1, reasoning: ranked.reasoning } : null;
    }).filter(Boolean) as { match: ProductMatch; rank: number; reasoning: string }[];
  };

  // Get other matches (excluding top 3)
  const getOtherMatches = (item: ItemWithMatches): ProductMatch[] => {
    return item.matches.filter(m => m.llm_rank === null);
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
        {items.map((item, index) => {
          const topMatches = getTopMatches(item);
          const otherMatches = getOtherMatches(item);
          
          return (
            <div key={index} className={styles.itemCard}>
              <div
                className={styles.itemHeader}
                onClick={() => toggleExpand(index)}
              >
                <div className={styles.itemInfo}>
                  <span className={styles.itemNumber}>#{index + 1}</span>
                  <div className={styles.itemText}>
                    <p className={styles.rawText}>{item.raw_text}</p>
                    <div className={styles.structuredQuery}>
                      {item.category && <span className={styles.queryFeature}>📦 {item.category}</span>}
                      {item.size && <span className={styles.queryFeature}>📏 {item.size}</span>}
                      {item.materials && <span className={styles.queryFeature}>🔩 {item.materials}</span>}
                      {item.standard && <span className={styles.queryFeature}>📋 {item.standard}</span>}
                      {item.dimensions && <span className={styles.queryFeature}>📐 {item.dimensions}</span>}
                      {item.brand && <span className={styles.queryFeature}>🏷️ {item.brand}</span>}
                      {item.application && <span className={styles.queryFeature}>🔧 {item.application}</span>}
                    </div>
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
                      {/* LLM Top 3 Matches - Highlighted */}
                      {topMatches.length > 0 && item.llmRerank && (
                        <div className={styles.bestMatchSection}>
                          <div className={styles.bestMatchHeader}>
                            <span className={styles.bestMatchIcon}>🎯</span>
                            <span className={styles.bestMatchTitle}>LLM Top Matches</span>
                            <span className={styles.confidence}>{item.llmRerank.confidence}</span>
                          </div>
                          
                          {topMatches.map(({ match, rank, reasoning }) => (
                            <div 
                              key={match.article_number} 
                              className={`${styles.bestMatchCard} ${rank === 1 ? styles.topRanked : ''}`}
                            >
                              <div className={styles.matchHeader}>
                                <span className={styles.rankBadge}>
                                  {getRankEmoji(rank)} #{rank}
                                </span>
                                <span className={styles.articleNumber}>
                                  {match.article_number}
                                </span>
                                <span className={styles.score}>
                                  {(match.similarity_score * 100).toFixed(1)}%
                                </span>
                              </div>
                              <p className={styles.headline}>{match.headline}</p>
                              <div className={styles.features}>
                                {match.category && <span className={styles.feature}>📦 {match.category}</span>}
                                {match.size && <span className={styles.feature}>📏 {match.size}</span>}
                                {match.materials && <span className={styles.feature}>🔩 {match.materials}</span>}
                                {match.standard && <span className={styles.feature}>📋 {match.standard}</span>}
                              </div>
                              <p className={styles.reasoning}>{reasoning}</p>
                            </div>
                          ))}
                        </div>
                      )}
                      
                      {/* Other Matches */}
                      {otherMatches.length > 0 && (
                        <>
                          <div className={styles.otherMatchesHeader}>
                            Other Matches ({otherMatches.length})
                          </div>
                          {otherMatches.slice(0, 5).map((match, mIndex) => (
                            <div key={mIndex} className={styles.matchCard}>
                              <div className={styles.matchHeader}>
                                <span className={styles.articleNumber}>
                                  {match.article_number}
                                </span>
                                <span className={styles.score}>
                                  {(match.similarity_score * 100).toFixed(1)}%
                                </span>
                              </div>
                              <p className={styles.headline}>{match.headline}</p>
                              <div className={styles.features}>
                                {match.category && <span className={styles.feature}>📦 {match.category}</span>}
                                {match.size && <span className={styles.feature}>📏 {match.size}</span>}
                                {match.materials && <span className={styles.feature}>🔩 {match.materials}</span>}
                              </div>
                            </div>
                          ))}
                        </>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
