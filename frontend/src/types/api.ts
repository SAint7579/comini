export interface SearchRequest {
  query: string;
  top_k: number;
  expand_abbreviations: boolean;
}

export interface ProductMatch {
  article_number: string;
  combined_description: string;
  long_description: string | null;
  image_url: string | null;
  similarity_score: number;
}

export interface SearchResponse {
  original_query: string;
  expanded_query: string;
  detected_abbreviations: string[];
  matches: ProductMatch[];
}

export interface HealthResponse {
  status: string;
  database: string;
}

