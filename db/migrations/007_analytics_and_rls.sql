-- db/migrations/007_analytics_and_rls.sql
-- Minimal analytics for router outcomes + a dashboard view (idempotent).
-- RLS templates included but commented out.

create table if not exists public.agent_decisions (
  id           uuid primary key default gen_random_uuid(),
  agent_slug   text not null,
  user_id      text,
  query_text   text not null,
  was_success  boolean,
  latency_ms   integer,
  extra        jsonb default '{}'::jsonb,
  created_at   timestamptz default now()
);

create index if not exists agent_decisions_created_idx on public.agent_decisions (created_at desc);
create index if not exists agent_decisions_agent_created_idx on public.agent_decisions (agent_slug, created_at desc);
create index if not exists agent_decisions_user_created_idx  on public.agent_decisions (user_id, created_at desc);

create or replace view public.v_agent_recent_outcomes as
select
  agent_slug,
  count(*)                                         as total,
  sum((was_success is true)::int)                  as successes,
  round(100.0 * sum((was_success is true)::int) / nullif(count(*),0), 1) as success_rate_pct,
  min(created_at) as first_seen,
  max(created_at) as last_seen
from public.agent_decisions
where created_at >= now() - interval '30 days'
group by agent_slug
order by total desc;

-- (Commented) enable RLS later
-- alter table public.agent_embeddings   enable row level security;
-- alter table public.agent_capabilities enable row level security;
-- alter table public.agent_decisions    enable row level security;

-- (Commented) example service-role policies
-- create policy "svc_all_embeddings"
--   on public.agent_embeddings for all to service_role
--   using (true) with check (true);
-- create policy "svc_all_capabilities"
--   on public.agent_capabilities for all to service_role
--   using (true) with check (true);
-- create policy "svc_all_decisions"
--   on public.agent_decisions for all to service_role
--   using (true) with check (true);
