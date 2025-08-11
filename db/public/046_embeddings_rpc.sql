-- db/public/046_embeddings_rpc.sql
create or replace function public.semantic_search_agent_embeddings(
  p_namespace text,
  p_query_embedding vector(1024),
  p_match_count int default 10
)
returns table(doc_id text, text text, score float4, metadata jsonb)
language sql stable as $$
  select
    ae.doc_id,
    ae.text,
    1 - (ae.embedding <=> p_query_embedding) as score,  -- cosine similarity
    ae.metadata
  from public.agent_embeddings ae
  where ae.namespace = p_namespace
  order by ae.embedding <=> p_query_embedding
  limit p_match_count
$$;
