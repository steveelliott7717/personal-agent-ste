-- db/migrations/005_fix_agent_embeddings_doc_id.sql
-- Ensure agent_embeddings has doc_id and the upsert key (namespace, doc_id)

-- Table safety (should already exist)
CREATE TABLE IF NOT EXISTS public.agent_embeddings (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  namespace  text NOT NULL,
  text       text NOT NULL,
  embedding  vector(1024),
  metadata   jsonb DEFAULT '{}'::jsonb,
  kind       text DEFAULT 'utterance' CHECK (kind IN ('utterance','capability')),
  ref        text,
  created_at timestamptz DEFAULT now()
);

-- Add doc_id if missing
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='public' AND table_name='agent_embeddings' AND column_name='doc_id'
  ) THEN
    ALTER TABLE public.agent_embeddings ADD COLUMN doc_id text;
  END IF;
END$$;

-- Backfill null doc_id for any existing rows (so the unique index can be created)
UPDATE public.agent_embeddings
SET doc_id = COALESCE(doc_id, gen_random_uuid()::text)
WHERE doc_id IS NULL;

-- Create the unique index used by on_conflict="namespace,doc_id"
CREATE UNIQUE INDEX IF NOT EXISTS ux_agent_embeddings_ns_doc
  ON public.agent_embeddings(namespace, doc_id);

-- Optional: nudge PostgREST to refresh schema (safe if pgrst channel exists)
-- SELECT pg_notify('pgrst', 'reload schema');
