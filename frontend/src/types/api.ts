export interface SearchRequest {
  query: string;
  top_k: number;
  expand_abbreviations: boolean;
  rerank?: boolean;
}

export interface ProductMatch {
  article_number: string;
  headline: string;
  long_description: string | null;
  image_url: string | null;
  
  // LLM-extracted features
  standard: string | null;
  materials: string | null;
  size: string | null;
  dimensions: string | null;
  category: string | null;
  brand: string | null;
  application: string | null;
  additional_info: string | null;
  
  similarity_score: number;
  llm_rank: number | null;  // 1, 2, or 3 if LLM selected this as top match
}

export interface LLMRankedMatch {
  index: number;
  article_number: string;
  reasoning: string;
}

export interface LLMRerankInfo {
  top_matches: LLMRankedMatch[];  // Top 3 matches with reasoning
  confidence: string;
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

// RFQ Types - Structured format matching product schema
export interface RFQItem {
  raw_text: string;
  search_query: string;
  quantity: number | null;
  unit: string | null;
  notes: string | null;
  
  // Structured fields (same as product schema)
  standard: string | null;
  materials: string | null;
  size: string | null;
  dimensions: string | null;
  category: string | null;
  brand: string | null;
  application: string | null;
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
