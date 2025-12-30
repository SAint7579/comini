-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Example table with vector column (you can modify/remove this)
-- CREATE TABLE items (
--   id SERIAL PRIMARY KEY,
--   embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
--   content TEXT,
--   metadata JSONB,
--   created_at TIMESTAMPTZ DEFAULT NOW()
-- );

-- Create an index for faster similarity search (uncomment when you have a table)
-- CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

