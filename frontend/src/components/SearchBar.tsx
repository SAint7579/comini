'use client';

import { useState, KeyboardEvent } from 'react';
import styles from './SearchBar.module.css';

interface SearchBarProps {
  onSearch: (query: string, topK: number, expandAbbreviations: boolean, rerank: boolean) => void;
  isLoading: boolean;
}

export default function SearchBar({ onSearch, isLoading }: SearchBarProps) {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(20);
  const [expandAbbreviations, setExpandAbbreviations] = useState(true);
  const [rerank, setRerank] = useState(true);

  const handleSearch = () => {
    if (query.trim()) {
      onSearch(query.trim(), topK, expandAbbreviations, rerank);
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.searchBox}>
        <input
          type="text"
          className={styles.input}
          placeholder="Search products... (e.g., TX25 bit gv zn)"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isLoading}
        />
        <button 
          className={styles.button} 
          onClick={handleSearch}
          disabled={isLoading || !query.trim()}
        >
          {isLoading ? 'Searching...' : 'Search'}
        </button>
      </div>
      
      <div className={styles.options}>
        <label className={styles.optionLabel}>
          <input
            type="checkbox"
            checked={expandAbbreviations}
            onChange={(e) => setExpandAbbreviations(e.target.checked)}
          />
          Expand abbreviations
        </label>
        
        <label className={styles.optionLabel}>
          <input
            type="checkbox"
            checked={rerank}
            onChange={(e) => setRerank(e.target.checked)}
          />
          LLM best match
        </label>
        
        <label className={styles.optionLabel}>
          <input
            type="number"
            className={styles.numberInput}
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value) || 20)}
            min={1}
            max={100}
          />
          Results
        </label>
      </div>
    </div>
  );
}

