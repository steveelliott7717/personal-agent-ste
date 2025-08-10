CREATE TABLE IF NOT EXISTS public.router_memory (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    text NOT NULL,
  query_text text NOT NULL,
  decision   text NOT NULL,  -- e.g. 'meals','clarify','none'
  rewrite    text,
  reason     text,
  response   text,
  options    jsonb,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.router_alias (
  id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    text NOT NULL,
  phrase     text NOT NULL,
  route      text NOT NULL,
  created_at timestamptz DEFAULT now(),
  UNIQUE (user_id, phrase)
);

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname='idx_router_memory_user_created') THEN
    CREATE INDEX idx_router_memory_user_created
      ON public.router_memory (user_id, created_at DESC);
  END IF;
END $$;
