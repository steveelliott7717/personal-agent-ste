-- ============================================
-- RMS GPT Baseline Schema
-- Creates tables & policies needed for repo search
-- ============================================

-- Drop old tables if they exist (optional for clean slate)
DROP TABLE IF EXISTS repo_files CASCADE;
DROP TABLE IF EXISTS repo_chunks CASCADE;

-- =======================
-- repo_files table
-- =======================
CREATE TABLE repo_files (
    id BIGSERIAL PRIMARY KEY,
    repo TEXT NOT NULL,
    branch TEXT NOT NULL,
    path TEXT NOT NULL,
    file_sha TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX ux_repo_file
    ON repo_files (repo, branch, path);

-- =======================
-- repo_chunks table
-- =======================
CREATE TABLE repo_chunks (
    id BIGSERIAL PRIMARY KEY,
    file_id BIGINT REFERENCES repo_files(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_sha TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX ux_repo_chunk
    ON repo_chunks (file_id, chunk_index);

CREATE INDEX idx_repo_chunks_embedding
    ON repo_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- =======================
-- Row Level Security
-- =======================
ALTER TABLE repo_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE repo_chunks ENABLE ROW LEVEL SECURITY;

-- Allow full access for service_role (your RMS processes)
CREATE POLICY "Allow all for service_role"
    ON repo_files
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow all for service_role"
    ON repo_chunks
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- =======================
-- Functions
-- =======================

-- Function for semantic search (1536-dim embeddings)
DROP FUNCTION IF EXISTS repo_search_1536(vector, text, text, text, integer);

CREATE FUNCTION repo_search_1536(
    query_embedding vector(1536),
    repo TEXT,
    branch TEXT,
    prefix TEXT,
    match_count INTEGER
)
RETURNS TABLE (
    id BIGINT,
    file_id BIGINT,
    content TEXT,
    similarity DOUBLE PRECISION,
    path TEXT,
    file_sha TEXT,
    chunk_sha TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.file_id,
        c.content,
        1 - (c.embedding <=> query_embedding) AS similarity,
        f.path,
        f.file_sha,
        c.chunk_sha
    FROM repo_chunks c
    JOIN repo_files f ON f.id = c.file_id
    WHERE f.repo = repo
      AND f.branch = branch
      AND f.path LIKE prefix || '%'
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- =======================
-- Indexing triggers
-- =======================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER set_updated_at_repo_files
BEFORE UPDATE ON repo_files
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER set_updated_at_repo_chunks
BEFORE UPDATE ON repo_chunks
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();
