-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Products table with LLM-extracted features
-- This table stores industrial tools/fasteners with structured metadata
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    
    -- Core identification
    article_number TEXT UNIQUE NOT NULL,
    headline TEXT NOT NULL,
    long_description TEXT,
    image_url TEXT,
    
    -- LLM-extracted structured features
    standard TEXT,           -- Industry standards (DIN, ISO, VDE, etc.)
    materials TEXT,          -- Materials and surface treatments
    size TEXT,               -- Size specifications (drive size, bit size, etc.)
    dimensions TEXT,         -- Physical dimensions (length, width, diameter, weight)
    category TEXT,           -- Product category/type
    brand TEXT,              -- Brand/manufacturer
    application TEXT,        -- Intended use and applications
    additional_info TEXT,    -- Other relevant information
    
    -- Vector embedding for semantic search
    embedding vector(1536),  -- text-embedding-3-small dimensions
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS products_embedding_idx 
ON products USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Create indexes on commonly queried fields
CREATE INDEX IF NOT EXISTS products_category_idx ON products (category);
CREATE INDEX IF NOT EXISTS products_brand_idx ON products (brand);
CREATE INDEX IF NOT EXISTS products_standard_idx ON products (standard);

-- Function to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger to update timestamp on row update
DROP TRIGGER IF EXISTS update_products_updated_at ON products;
CREATE TRIGGER update_products_updated_at
    BEFORE UPDATE ON products
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
