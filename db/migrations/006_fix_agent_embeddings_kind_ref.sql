-- db/migrations/006_fix_agent_embeddings_kind_ref.sql
-- Ensure agent_embeddings has kind + ref columns aligned with app logic.

CREATE TABLE IF NOT EXISTS public.agent_embeddings (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  namespace  text NOT NULL,
  text       text NOT NULL,
  embedding  vector(1024),
  metadata   jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz DEFAULT now()
);

-- kind column with default + constraint
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='public' AND table_name='agent_embeddings' AND column_name='kind'
  ) THEN
    ALTER TABLE public.agent_embeddings ADD COLUMN kind text;
  END IF;

  -- Default
  BEGIN
    ALTER TABLE public.agent_embeddings ALTER COLUMN kind SET DEFAULT 'utterance';
  EXCEPTION WHEN undefined_column THEN
    -- ignore
  END;

  -- Constraint (create if missing)
  IF NOT EXISTS (
    SELECT 1
    FROM information_schema.constraint_column_usage ccu
    WHERE ccu.table_schema='public' AND ccu.table_name='agent_embeddings' AND ccu.constraint_name='agent_embeddings_kind_chk'
  ) THEN
    ALTER TABLE public.agent_embeddings
      ADD CONSTRAINT agent_embeddings_kind_chk CHECK (kind IN ('utterance','capability'));
  END IF;

  -- Backfill nulls to default
  UPDATE public.agent_embeddings SET kind='utterance' WHERE kind IS NULL;
END$$;

-- ref column
ALTER TABLE public.agent_embeddings
  ADD COLUMN IF NOT EXISTS ref text;

-- Keep the unique index (namespace, doc_id)
CREATE UNIQUE INDEX IF NOT EXISTS ux_agent_embeddings_ns_doc
  ON public.agent_embeddings(namespace, doc_id);
