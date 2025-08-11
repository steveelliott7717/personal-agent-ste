-- db/migrations/002_embeddings_and_capabilities.sql
-- Idempotent setup for embeddings, capabilities, retrieval corpus, and RPC.

-- 1) Ensure pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2) Ensure public.agent_embeddings has the right shape for Cohere 1024-d + metadata + routing flags
DO $$
BEGIN
  -- Create table if it doesn't exist (minimal columns; we'll add/align below)
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema='public' AND table_name='agent_embeddings'
  ) THEN
    CREATE TABLE public.agent_embeddings (
      id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      namespace  text NOT NULL,
      doc_id     text,
      text       text NOT NULL,
      embedding  vector(1024),
      metadata   jsonb DEFAULT '{}'::jsonb,
      kind       text DEFAULT 'utterance' CHECK (kind IN ('utterance','capability')),
      ref        text,
      created_at timestamptz DEFAULT now()
    );
  END IF;

  -- Add doc_id if missing
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='public' AND table_name='agent_embeddings' AND column_name='doc_id'
  ) THEN
    ALTER TABLE public.agent_embeddings ADD COLUMN doc_id text;
  END IF;

  -- Rename meta -> metadata (older file used "meta")
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='public' AND table_name='agent_embeddings' AND column_name='meta'
  ) THEN
    ALTER TABLE public.agent_embeddings RENAME COLUMN meta TO metadata;
  END IF;

  -- Ensure embedding is vector(1024)
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='public' AND table_name='agent_embeddings' AND column_name='embedding'
      AND udt_name <> 'vector'
  ) THEN
    ALTER TABLE public.agent_embeddings DROP COLUMN embedding;
    ALTER TABLE public.agent_embeddings ADD COLUMN embedding vector(1024);
  ELSIF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='public' AND table_name='agent_embeddings' AND column_name='embedding'
  ) THEN
    ALTER TABLE public.agent_embeddings ADD COLUMN embedding vector(1024);
  END IF;

  -- Ensure metadata exists
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='public' AND table_name='agent_embeddings' AND column_name='metadata'
  ) THEN
    ALTER TABLE public.agent_embeddings ADD COLUMN metadata jsonb DEFAULT '{}'::jsonb;
  END IF;

  -- Add kind column w/ constraint & default
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='public' AND table_name='agent_embeddings' AND column_name='kind'
  ) THEN
    ALTER TABLE public.agent_embeddings ADD COLUMN kind text;
    ALTER TABLE public.agent_embeddings ALTER COLUMN kind SET DEFAULT 'utterance';
    ALTER TABLE public.agent_embeddings
      ADD CONSTRAINT agent_embeddings_kind_chk CHECK (kind IN ('utterance','capability'));
  END IF;

  -- Add ref (e.g., agent slug) column
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='public' AND table_name='agent_embeddings' AND column_name='ref'
  ) THEN
    ALTER TABLE public.agent_embeddings ADD COLUMN ref text;
  END IF;
END$$;

-- Upsert key used by code
CREATE UNIQUE INDEX IF NOT EXISTS ux_agent_embeddings_ns_doc
  ON public.agent_embeddings(namespace, doc_id);

-- Vector index
CREATE INDEX IF NOT EXISTS agent_embeddings_vec_idx
  ON public.agent_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 3) Capabilities table (one row per agent capability concept; start simple: one row per agent)
CREATE TABLE IF NOT EXISTS public.agent_capabilities (
  agent_slug    text PRIMARY KEY,
  description   text NOT NULL,
  param_schema  jsonb DEFAULT '{}'::jsonb,
  embedding     vector(1024),
  created_at    timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS agent_capabilities_vec_idx
  ON public.agent_capabilities USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- 4) A view that blends capabilities with historical utterances (router memory)
--    Keep simple columns useful for retrieval/debug.
CREATE OR REPLACE VIEW public.router_retrieval_corpus AS
SELECT
  'capability'::text AS source,
  ac.agent_slug     AS ref,
  'routing'::text   AS namespace,
  ac.agent_slug     AS doc_id,
  ac.description    AS text,
  ac.embedding      AS embedding,
  '{}'::jsonb       AS metadata,
  now()             AS created_at
FROM public.agent_capabilities ac
UNION ALL
SELECT
  'utterance'::text AS source,
  ae.ref            AS ref,
  ae.namespace      AS namespace,
  ae.doc_id         AS doc_id,
  ae.text           AS text,
  ae.embedding      AS embedding,
  ae.metadata       AS metadata,
  ae.created_at     AS created_at
FROM public.agent_embeddings ae
WHERE (ae.kind IS NULL OR ae.kind = 'utterance');

-- 5) RPC for semantic search (recreate to be safe). Matches your retriever signature.
CREATE OR REPLACE FUNCTION public.semantic_search_agent_embeddings(
  p_namespace text,
  p_query_embedding vector(1024),
  p_match_count int DEFAULT 10
)
RETURNS TABLE(doc_id text, text text, score float4, metadata jsonb)
LANGUAGE sql STABLE AS $$
  SELECT
    ae.doc_id,
    ae.text,
    1 - (ae.embedding <=> p_query_embedding) AS score,  -- cosine similarity
    ae.metadata
  FROM public.agent_embeddings ae
  WHERE ae.namespace = p_namespace
  ORDER BY ae.embedding <=> p_query_embedding
  LIMIT p_match_count
$$;
