\set ON_ERROR_STOP on
\i db/public/000_extensions.sql
\i db/public/010_registry.sql
\i db/public/020_router.sql
\i db/public/030_data_bus.sql
\i db/public/040_embeddings.sql
\i db/public/045_embeddings_fix.sql
\i db/public/046_embeddings_rpc.sql

\i db/meals/000_schema.sql
\i db/meals/010_tables.sql
\i db/meals/020_indexes.sql
-- \i db/meals/030_policies.sql  -- enable when ready

\i db/seed.sql
