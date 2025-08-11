-- Safe, idempotent test migration
create extension if not exists pgcrypto;

create table if not exists public.ci_probe (
  id uuid primary key default gen_random_uuid(),
  note text,
  created_at timestamptz default now()
);
