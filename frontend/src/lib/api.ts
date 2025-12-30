import { SearchRequest, SearchResponse, HealthResponse, RFQUploadResponse } from '@/types/api';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function searchProducts(request: SearchRequest): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Search failed');
  }

  return response.json();
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  
  if (!response.ok) {
    throw new Error('Health check failed');
  }

  return response.json();
}

export async function uploadRFQ(file: File): Promise<RFQUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/upload-rfq`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'RFQ upload failed');
  }

  return response.json();
}

export async function searchForRFQItem(query: string, topK: number = 20): Promise<SearchResponse> {
  return searchProducts({
    query,
    top_k: topK,
    expand_abbreviations: false, // Already expanded by RFQ processing
    rerank: true, // Enable LLM reranking to find best match
  });
}

