create extension if not exists vector;

-- If table exists, add missing cols / rename; else create it aligned to code
do $$
begin
  if not exists (
    select 1 from information_schema.columns
    where table_schema='public' and table_name='agent_embeddings' and column_name='doc_id'
  ) then
    alter table public.agent_embeddings add column doc_id text;
  end if;

  -- rename meta -> metadata to match Python
  if exists (
    select 1 from information_schema.columns
    where table_schema='public' and table_name='agent_embeddings' and column_name='meta'
  ) then
    alter table public.agent_embeddings rename column meta to metadata;
  end if;

  -- switch embedding to vector(1024)
  -- if you already have data in bytea/json, you can backfill later; this just adds the column if missing
  if not exists (
    select 1 from information_schema.columns
    where table_schema='public' and table_name='agent_embeddings' and column_name='embedding'
      and udt_name = 'vector'
  ) then
    alter table public.agent_embeddings drop column if exists embedding;
    alter table public.agent_embeddings add column embedding vector(1024);
  end if;
end$$;

-- upsert key your code uses
create unique index if not exists ux_agent_embeddings_ns_doc
  on public.agent_embeddings(namespace, doc_id);

-- vector index for fast search
create index if not exists agent_embeddings_vec_idx
  on public.agent_embeddings using ivfflat (embedding vector_cosine_ops) with (lists = 100);
