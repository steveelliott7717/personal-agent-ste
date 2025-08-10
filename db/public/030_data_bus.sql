-- Generic events table for cross-agent comms
CREATE TABLE IF NOT EXISTS public.agent_events (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  ts           timestamptz DEFAULT now(),
  agent_slug   text NOT NULL,
  event_type   text NOT NULL,         -- 'meal_completed','stock_low',...
  entity_ref   jsonb DEFAULT '{}'::jsonb, -- {schema:'meals', table:'meal_plan', id:'...'}
  payload      jsonb DEFAULT '{}'::jsonb, -- arbitrary details
  processed_by jsonb DEFAULT '[]'::jsonb  -- array of consumer ids that handled it
);

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname='idx_agent_events_slug_ts') THEN
    CREATE INDEX idx_agent_events_slug_ts
      ON public.agent_events (agent_slug, ts DESC);
  END IF;
END $$;
