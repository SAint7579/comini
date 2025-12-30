export interface SearchRequest {
  query: string;
  top_k: number;
  expand_abbreviations: boolean;
  rerank?: boolean;
}

export interface ProductMatch {
  article_number: string;
  combined_description: string;
  long_description: string | null;
  image_url: string | null;
  similarity_score: number;
  is_llm_best_match: boolean;
}

export interface LLMRerankInfo {
  best_match_index: number;
  best_match_article: string;
  confidence: string;
  reasoning: string;
}

export interface SearchResponse {
  original_query: string;
  expanded_query: string;
  detected_abbreviations: string[];
  matches: ProductMatch[];
  llm_rerank: LLMRerankInfo | null;
}

export interface HealthResponse {
  status: string;
  database: string;
}

// RFQ Types
export interface RFQItem {
  raw_text: string;
  search_query: string;
  quantity: number | null;
  unit: string | null;
  notes: string | null;
}

export interface RFQUploadResponse {
  source_type: string;
  total_items: number;
  chunks_processed: number;
  items: RFQItem[];
}

export interface RFQFileEntry {
  id: string;
  filename: string;
  uploadedAt: Date;
  sourceType: string;
  totalItems: number;
  items: RFQItem[];
}

export interface RFQItemWithMatches extends RFQItem {
  matches?: ProductMatch[];
  isLoading?: boolean;
  error?: string;
}

