import io
import asyncio
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_utils.data_preprocessing import process_products
from rag_utils.embedding_utils import (
    init_db,
    generate_embeddings_batched,
    generate_query_embedding,
    store_products_with_embeddings,
    get_db_connection,
    check_db_health,
)
from rag_utils.query_utils import expand_query_async, ExpandedQuery, rerank_results_async, RerankResult
from document_utils.rfq_processing import (
    process_rfq_file_async,
    RFQItem,
    RFQProcessingResult,
)
from std_utils import get_logger

logger = get_logger("comini.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_db()
    yield


app = FastAPI(
    title="Comini Product RAG API",
    description="API for processing product data and storing embeddings",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessingResponse(BaseModel):
    message: str
    products_processed: int
    products_stored: int


@app.post("/upload-csv", response_model=ProcessingResponse)
async def upload_and_process_csv(
    file: UploadFile = File(...),
    fetch_images: bool = False
):
    """
    Upload a CSV file, process it, generate embeddings, and store in database.
    
    Args:
        file: CSV file with product data
        fetch_images: Whether to fetch image URLs from product pages (slow)
    
    Returns:
        Processing summary with counts
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate required columns
        required_columns = ['Headline', 'Short Description', 'Long Description', 'Article Number', 'URL']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing}"
            )
        
        # Step 1: Process products
        logger.info(f"[1/3] Processing {len(df)} products from CSV...")
        processed_df = await asyncio.to_thread(
            process_products, df, fetch_images=fetch_images
        )
        
        # Filter out rows with empty descriptions
        processed_df = processed_df[
            processed_df['combined_description'].notna() & 
            (processed_df['combined_description'].str.strip() != '')
        ]
        logger.info(f"[1/3] Processing complete. {len(processed_df)} valid products")
        
        if processed_df.empty:
            raise HTTPException(status_code=400, detail="No valid products to process")
        
        # Step 2: Generate embeddings in batches (async, parallelized)
        descriptions = processed_df['combined_description'].tolist()
        logger.info(f"[2/3] Generating embeddings for {len(descriptions)} products...")
        embeddings = await asyncio.to_thread(
            generate_embeddings_batched, descriptions
        )
        logger.info(f"[2/3] Embeddings complete. Generated {len(embeddings)} vectors")
        
        # Step 3: Store in database (run in thread)
        logger.info("[3/3] Storing products in database...")
        stored_count = await asyncio.to_thread(
            store_products_with_embeddings, processed_df, embeddings
        )
        logger.info(f"[3/3] Database insert complete. Stored {stored_count} products")
        
        return ProcessingResponse(
            message="Products processed and stored successfully",
            products_processed=len(df),
            products_stored=stored_count
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    expand_abbreviations: bool = True  # Set to False to skip LLM expansion
    rerank: bool = True  # Use LLM to pick the best match from top_k


class ProductMatch(BaseModel):
    article_number: str
    combined_description: str
    long_description: str | None
    image_url: str | None
    similarity_score: float
    is_llm_best_match: bool = False  # True if LLM selected this as best match


class LLMRerankInfo(BaseModel):
    """Information about the LLM reranking result."""
    best_match_index: int
    best_match_article: str
    confidence: str
    reasoning: str


class SearchResponse(BaseModel):
    original_query: str
    expanded_query: str
    detected_abbreviations: list[str]
    matches: list[ProductMatch]
    llm_rerank: LLMRerankInfo | None = None  # Populated if rerank=True


@app.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """
    Search for products using semantic similarity with optional LLM reranking.
    
    The query is first expanded using an LLM to handle abbreviations,
    then matched against product embeddings using cosine similarity.
    Optionally, an LLM reranks the results to pick the best match.
    
    Args:
        request: Search query, number of results (default 20), and options
    
    Returns:
        Expanded query info, top matching products, and LLM's best pick
    """
    logger.info(f"Search request: {request.query} (expand={request.expand_abbreviations}, rerank={request.rerank})")
    
    try:
        # Step 1: Expand query using LLM (optional)
        if request.expand_abbreviations:
            logger.info("[1/4] Expanding query with LLM...")
            expanded = await expand_query_async(request.query)
            search_query = expanded.expanded_query
            detected_abbrevs = expanded.detected_abbreviations
            logger.info(f"[1/4] Expanded: '{search_query}'")
        else:
            logger.info("[1/4] Skipping query expansion (disabled)")
            search_query = request.query
            detected_abbrevs = []
        
        # Step 2: Generate embedding for search query
        logger.info("[2/4] Generating query embedding...")
        query_embedding = await asyncio.to_thread(
            generate_query_embedding, search_query
        )
        logger.info("[2/4] Query embedding generated")
        
        # Step 3: Search database using cosine similarity
        logger.info(f"[3/4] Searching for top {request.top_k} matches...")
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Cosine similarity: 1 - cosine_distance
        # pgvector's <=> operator returns cosine distance
        cur.execute("""
            SELECT 
                article_number,
                combined_description,
                long_description,
                image_url,
                1 - (embedding <=> %s::vector) as similarity_score
            FROM products
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, request.top_k))
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        logger.info(f"[3/4] Found {len(results)} matches")
        
        # Step 4: LLM Reranking (optional)
        llm_rerank_info = None
        best_match_index = -1
        
        if request.rerank and len(results) > 0:
            logger.info("[4/4] Reranking with LLM...")
            
            # Prepare products for reranking
            products_for_rerank = [
                {
                    "article_number": row[0],
                    "combined_description": row[1],
                }
                for row in results
            ]
            
            rerank_result = await rerank_results_async(
                request.query,  # Use original query for reranking
                products_for_rerank
            )
            
            best_match_index = rerank_result.best_match_index
            llm_rerank_info = LLMRerankInfo(
                best_match_index=best_match_index,
                best_match_article=results[best_match_index][0],
                confidence=rerank_result.confidence,
                reasoning=rerank_result.reasoning,
            )
            
            logger.info(f"[4/4] LLM picked index {best_match_index}: {results[best_match_index][0]}")
        else:
            logger.info("[4/4] Skipping reranking (disabled or no results)")
        
        # Build response with LLM best match flag
        matches = [
            ProductMatch(
                article_number=row[0],
                combined_description=row[1],
                long_description=row[2],
                image_url=row[3],
                similarity_score=round(row[4], 4),
                is_llm_best_match=(i == best_match_index)
            )
            for i, row in enumerate(results)
        ]
        
        return SearchResponse(
            original_query=request.query,
            expanded_query=search_query,
            detected_abbreviations=detected_abbrevs,
            matches=matches,
            llm_rerank=llm_rerank_info,
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RFQ Processing Endpoints
# ============================================================================

class RFQItemResponse(BaseModel):
    """Single RFQ item in API response."""
    raw_text: str
    search_query: str
    quantity: int | None
    unit: str | None
    notes: str | None


class RFQUploadResponse(BaseModel):
    """Response from RFQ file upload."""
    source_type: str
    total_items: int
    chunks_processed: int
    items: list[RFQItemResponse]


@app.post("/upload-rfq", response_model=RFQUploadResponse)
async def upload_rfq(
    file: UploadFile = File(...),
    chunk_size: int | None = None
):
    """
    Upload an RFQ file (CSV, Excel, or PDF) and extract structured search queries.
    
    The file is processed in chunks for large files, with each item
    converted to an optimized search query with abbreviations expanded.
    
    For PDFs, the document is first converted to text using marker, then
    processed in text chunks.
    
    Args:
        file: RFQ file (CSV, XLSX, XLS, PDF)
        chunk_size: Rows per chunk (CSV/Excel) or chars per chunk (PDF).
                    Defaults: 50 rows for tabular, 4000 chars for PDF.
    
    Returns:
        Extracted items with raw text and search queries
    """
    # Validate file type
    filename = file.filename.lower()
    if filename.endswith('.csv'):
        file_type = 'csv'
    elif filename.endswith('.xlsx'):
        file_type = 'xlsx'
    elif filename.endswith('.xls'):
        file_type = 'xls'
    elif filename.endswith('.pdf'):
        file_type = 'pdf'
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported: CSV, XLSX, XLS, PDF"
        )
    
    try:
        logger.info(f"Processing RFQ upload: {file.filename} ({file_type})")
        
        # Read file contents
        contents = await file.read()
        file_obj = io.BytesIO(contents)
        
        # Process RFQ with async chunked processing
        result = await process_rfq_file_async(
            file_obj,
            file_type=file_type,
            chunk_size=chunk_size
        )
        
        logger.info(f"RFQ processed: {result.total_items} items from {result.chunks_processed} chunks")
        
        # Convert to response model
        return RFQUploadResponse(
            source_type=result.source_type,
            total_items=result.total_items,
            chunks_processed=result.chunks_processed,
            items=[
                RFQItemResponse(
                    raw_text=item.raw_text,
                    search_query=item.search_query,
                    quantity=item.quantity,
                    unit=item.unit,
                    notes=item.notes
                )
                for item in result.items
            ]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"RFQ processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check API and database health."""
    if check_db_health():
        return {"status": "healthy", "database": "connected"}
    else:
        return {"status": "unhealthy", "database": "disconnected"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
