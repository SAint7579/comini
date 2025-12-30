"""
Embedding utilities for product RAG system.

This module handles database connections, embedding generation,
and product storage/retrieval with vector similarity search.
"""

import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from std_utils import get_logger

# Initialize logger
logger = get_logger("comini.rag")

# Database configuration
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "comini_db"),
    "user": os.getenv("POSTGRES_USER", "comini"),
    "password": os.getenv("POSTGRES_PASSWORD", "comini_secret"),
}

# OpenAI Embeddings (text-embedding-3-small = 1536 dimensions)
# Using small model to stay under pgvector's 2000 dimension limit for indexes
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


@dataclass
class ProductSearchResult:
    """Result from a product similarity search."""
    article_number: str
    headline: str
    long_description: str | None
    image_url: str | None
    standard: str | None
    materials: str | None
    size: str | None
    dimensions: str | None
    category: str | None
    brand: str | None
    application: str | None
    additional_info: str | None
    similarity: float


def get_db_connection():
    """Create a database connection."""
    logger.debug(f"Connecting to database at {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    """Initialize the database table for products."""
    logger.info("Initializing database...")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    logger.info("Creating products table with LLM-extracted feature columns")
    # Create products table with all LLM-extracted feature columns
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS products (
            id SERIAL PRIMARY KEY,
            
            -- Core identification
            article_number TEXT UNIQUE NOT NULL,
            headline TEXT NOT NULL,
            long_description TEXT,
            image_url TEXT,
            
            -- LLM-extracted structured features
            standard TEXT,
            materials TEXT,
            size TEXT,
            dimensions TEXT,
            category TEXT,
            brand TEXT,
            application TEXT,
            additional_info TEXT,
            
            -- Vector embedding for semantic search
            embedding vector({EMBEDDING_DIMENSIONS}),
            
            -- Metadata
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    
    # Create HNSW index (supports higher dimensions than IVFFlat)
    logger.info("Creating HNSW index for similarity search")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS products_embedding_idx 
        ON products USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    
    # Create indexes on commonly queried fields
    cur.execute("CREATE INDEX IF NOT EXISTS products_category_idx ON products (category);")
    cur.execute("CREATE INDEX IF NOT EXISTS products_brand_idx ON products (brand);")
    cur.execute("CREATE INDEX IF NOT EXISTS products_standard_idx ON products (standard);")
    
    conn.commit()
    cur.close()
    conn.close()
    
    logger.info("Database initialization complete")


def get_embeddings_model() -> OpenAIEmbeddings:
    """Get the OpenAI embeddings model instance."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    embeddings_model = get_embeddings_model()
    return embeddings_model.embed_documents(texts)


def generate_embeddings_batched(
    texts: list[str],
    batch_size: int = 100,
    max_workers: int = 4
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts in parallel batches.
    
    This function splits the texts into batches and processes them
    concurrently using a thread pool for better performance with
    large datasets.
    
    Args:
        texts: List of text strings to embed
        batch_size: Number of texts per batch (OpenAI recommends ~100)
        max_workers: Number of parallel threads
        
    Returns:
        List of embedding vectors in the same order as input
    """
    if not texts:
        return []
    
    # Split texts into batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    # Pre-allocate results list
    all_embeddings = [None] * len(texts)
    
    def embed_batch(batch_idx: int, batch: list[str]) -> tuple[int, list[list[float]]]:
        """Embed a single batch and return with its index."""
        embeddings_model = get_embeddings_model()
        embeddings = embeddings_model.embed_documents(batch)
        return batch_idx, embeddings
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(embed_batch, idx, batch): idx
            for idx, batch in enumerate(batches)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding batches"):
            batch_idx, embeddings = future.result()
            
            # Place embeddings in correct position
            start_idx = batch_idx * batch_size
            for i, emb in enumerate(embeddings):
                all_embeddings[start_idx + i] = emb
    
    return all_embeddings


def generate_query_embedding(query: str) -> list[float]:
    """
    Generate embedding for a single query text.
    
    Args:
        query: Query string to embed
        
    Returns:
        Embedding vector
    """
    embeddings_model = get_embeddings_model()
    return embeddings_model.embed_query(query)


def build_embedding_text(row: pd.Series) -> str:
    """
    Build the text to embed from product features.
    
    Combines headline, long description, and extracted features
    into a single searchable text.
    
    Args:
        row: DataFrame row with product features
        
    Returns:
        Combined text for embedding
    """
    parts = []
    
    # Core text
    if pd.notna(row.get('headline')):
        parts.append(row['headline'])
    if pd.notna(row.get('long_description')):
        parts.append(row['long_description'])
    
    # LLM-extracted features
    feature_fields = [
        ('Standard', 'standard'),
        ('Materials', 'materials'),
        ('Size', 'size'),
        ('Dimensions', 'dimensions'),
        ('Category', 'category'),
        ('Brand', 'brand'),
        ('Application', 'application'),
        ('Additional', 'additional_info'),
    ]
    
    for label, field in feature_fields:
        value = row.get(field)
        if pd.notna(value) and str(value).strip():
            parts.append(f"{label}: {value}")
    
    return ' '.join(parts)


def store_products_with_embeddings(
    processed_df: pd.DataFrame,
    embeddings: list[list[float]]
) -> int:
    """
    Store products with their embeddings in the database.
    
    Args:
        processed_df: DataFrame with processed product data (with LLM-extracted features)
        embeddings: List of embedding vectors matching the DataFrame rows
        
    Returns:
        Number of products stored
    """
    logger.info(f"Starting database insert for {len(processed_df)} products")
    
    conn = get_db_connection()
    cur = conn.cursor()
    logger.debug("Database connection established")
    
    # Prepare data for insertion
    logger.info("Preparing records for insertion...")
    records = []
    for idx, row in processed_df.iterrows():
        records.append((
            row['article_number'],
            row['headline'],
            row.get('long_description'),
            row.get('image_url'),
            row.get('standard'),
            row.get('materials'),
            row.get('size'),
            row.get('dimensions'),
            row.get('category'),
            row.get('brand'),
            row.get('application'),
            row.get('additional_info'),
            embeddings[processed_df.index.get_loc(idx)]
        ))
    logger.info(f"Prepared {len(records)} records")
    
    # Upsert products (update if article_number exists)
    logger.info("Executing upsert query...")
    insert_query = """
        INSERT INTO products (
            article_number, headline, long_description, image_url,
            standard, materials, size, dimensions, category, brand, application, additional_info,
            embedding
        )
        VALUES %s
        ON CONFLICT (article_number) 
        DO UPDATE SET 
            headline = EXCLUDED.headline,
            long_description = EXCLUDED.long_description,
            image_url = EXCLUDED.image_url,
            standard = EXCLUDED.standard,
            materials = EXCLUDED.materials,
            size = EXCLUDED.size,
            dimensions = EXCLUDED.dimensions,
            category = EXCLUDED.category,
            brand = EXCLUDED.brand,
            application = EXCLUDED.application,
            additional_info = EXCLUDED.additional_info,
            embedding = EXCLUDED.embedding,
            updated_at = NOW()
    """
    
    execute_values(
        cur, insert_query, records,
        template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)"
    )
    logger.info("Upsert query executed successfully")
    
    logger.info("Committing transaction...")
    conn.commit()
    stored_count = len(records)
    
    cur.close()
    conn.close()
    
    logger.info(f"Database insert complete. Stored {stored_count} products")
    return stored_count


def search_similar_products(
    query: str,
    top_k: int = 5
) -> list[ProductSearchResult]:
    """
    Search for similar products using semantic search.
    
    Args:
        query: Search query string
        top_k: Number of results to return
        
    Returns:
        List of ProductSearchResult objects
    """
    # Generate embedding for query
    query_embedding = generate_query_embedding(query)
    
    # Search in database
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            article_number,
            headline,
            long_description,
            image_url,
            standard,
            materials,
            size,
            dimensions,
            category,
            brand,
            application,
            additional_info,
            1 - (embedding <=> %s::vector) as similarity
        FROM products
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (query_embedding, query_embedding, top_k))
    
    results = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return [
        ProductSearchResult(
            article_number=row[0],
            headline=row[1],
            long_description=row[2],
            image_url=row[3],
            standard=row[4],
            materials=row[5],
            size=row[6],
            dimensions=row[7],
            category=row[8],
            brand=row[9],
            application=row[10],
            additional_info=row[11],
            similarity=row[12]
        )
        for row in results
    ]


def check_db_health() -> bool:
    """
    Check if the database connection is healthy.
    
    Returns:
        True if healthy, False otherwise
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()
        return True
    except Exception:
        return False
