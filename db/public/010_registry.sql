CREATE TABLE IF NOT EXISTS public.agents (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  slug          text UNIQUE NOT NULL,
  title         text NOT NULL,
  description   text,
  module_path   text,
  callable_name text,
  namespaces    jsonb DEFAULT '[]'::jsonb,
  capabilities  jsonb DEFAULT '[]'::jsonb,
  status        text DEFAULT 'enabled',
  version       text,
  created_at    timestamptz DEFAULT now(),
  updated_at    timestamptz DEFAULT now()
);

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes WHERE schemaname='public' AND indexname='idx_agents_slug_unique'
  ) THEN
    CREATE UNIQUE INDEX idx_agents_slug_unique ON public.agents (slug);
  END IF;
END$$;

CREATE TABLE IF NOT EXISTS public.agent_instructions (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_name   text NOT NULL, -- matches agents.slug
  tag          text NOT NULL, -- 'core','planning','logging', etc.
  instructions text NOT NULL,
  created_at   timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.agent_settings (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_slug      text NOT NULL,
  default_tables  jsonb DEFAULT '[]'::jsonb,
  instruction_tags jsonb DEFAULT '[]'::jsonb,
  post_hooks      jsonb DEFAULT '[]'::jsonb,
  created_at      timestamptz DEFAULT now()
);
