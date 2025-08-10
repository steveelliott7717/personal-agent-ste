-- If you store embeddings for routing or few-shot memory
-- Change dimension to match your embedding model
CREATE TABLE IF NOT EXISTS public.agent_embeddings (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  namespace  text NOT NULL,      -- 'routing','meals',...
  text       text NOT NULL,
  -- If using pgvector: CREATE EXTENSION IF NOT EXISTS vector; then:
  -- embedding vector(1536),
  embedding  bytea,              -- portable fallback if no pgvector
  meta       jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now()
);
