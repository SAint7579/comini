import io
import asyncio
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from rag_utils.data_preprocessing import process_products
from rag_utils.embedding_utils import (
    init_db,
    generate_embeddings_batched,
    store_products_with_embeddings,
    check_db_health,
)


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
        
        # Process products (run in thread to not block event loop)
        processed_df = await asyncio.to_thread(
            process_products, df, fetch_images=fetch_images
        )
        
        # Filter out rows with empty descriptions
        processed_df = processed_df[
            processed_df['combined_description'].notna() & 
            (processed_df['combined_description'].str.strip() != '')
        ]
        
        if processed_df.empty:
            raise HTTPException(status_code=400, detail="No valid products to process")
        
        # Generate embeddings in batches (async, parallelized)
        descriptions = processed_df['combined_description'].tolist()
        print(f"Generating embeddings for {len(descriptions)} products...")
        embeddings = await asyncio.to_thread(
            generate_embeddings_batched, descriptions
        )
        
        # Store in database (run in thread)
        stored_count = await asyncio.to_thread(
            store_products_with_embeddings, processed_df, embeddings
        )
        
        return ProcessingResponse(
            message="Products processed and stored successfully",
            products_processed=len(df),
            products_stored=stored_count
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except Exception as e:
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
