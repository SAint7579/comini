'use client';

import { useState } from 'react';
import SearchBar from '@/components/SearchBar';
import ExpansionInfo from '@/components/ExpansionInfo';
import ResultsList from '@/components/ResultsList';
import LLMBestMatch from '@/components/LLMBestMatch';
import { searchProducts } from '@/lib/api';
import { SearchResponse } from '@/types/api';
import styles from './page.module.css';

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (query: string, topK: number, expandAbbreviations: boolean, rerank: boolean) => {
    setIsLoading(true);
    setError(null);
    setHasSearched(true);

    try {
      const result = await searchProducts({
        query,
        top_k: topK,
        expand_abbreviations: expandAbbreviations,
        rerank,
      });
      setSearchResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setSearchResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className={styles.main}>
      <div className={styles.container}>
        <header className={styles.header}>
          <h1 className={styles.title}>Product Search</h1>
          <p className={styles.tagline}>Search the industrial tools & fasteners catalog</p>
        </header>

        <SearchBar onSearch={handleSearch} isLoading={isLoading} />

        {error && (
          <div className={styles.error}>
            Error: {error}
          </div>
        )}

        {isLoading && (
          <div className={styles.loading}>
            <div className={styles.spinner} />
            <p>Searching products...</p>
          </div>
        )}

        {!isLoading && searchResult && (
          <>
            <ExpansionInfo
              originalQuery={searchResult.original_query}
              expandedQuery={searchResult.expanded_query}
              abbreviations={searchResult.detected_abbreviations}
            />
            {searchResult.llm_rerank && (
              <LLMBestMatch 
                rerank={searchResult.llm_rerank}
                matches={searchResult.matches}
              />
            )}
            <ResultsList results={searchResult.matches} />
          </>
        )}

        {!isLoading && !searchResult && !hasSearched && (
          <div className={styles.emptyState}>
            <div className={styles.emptyIcon}>🔧</div>
            <p>Enter a search query to find products</p>
          </div>
        )}
      </div>
    </main>
  );
}

