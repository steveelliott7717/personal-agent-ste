-- db/migrations/zzz_probe.sql
-- Harmless probe to verify Actions runs and Supabase applies migrations.

CREATE TABLE IF NOT EXISTS public.__ci_probe (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  note text,
  created_at timestamptz DEFAULT now()
);

-- Optional no-op select:
-- SELECT now();
