-- scripts/schema_per_table.sql
-- Usage (psql): \set schema 'public' \set table 'events' \i scripts/schema_per_table.sql
WITH cols AS (
  SELECT
    c.table_schema, c.table_name,
    jsonb_agg(
      jsonb_build_object(
        'name', c.column_name,
        'data_type', c.data_type,
        'udt_name', c.udt_name,
        'is_nullable', (c.is_nullable='YES'),
        'default', c.column_default,
        'identity', c.is_identity,
        'ordinal', c.ordinal_position
      )
      ORDER BY c.ordinal_position
    ) AS columns
  FROM information_schema.columns c
  WHERE c.table_schema = :'schema' AND c.table_name = :'table'
  GROUP BY c.table_schema, c.table_name
),
pk AS (
  SELECT tc.table_schema, tc.table_name,
         jsonb_agg(kcu.column_name ORDER BY kcu.ordinal_position) AS primary_key
  FROM information_schema.table_constraints tc
  JOIN information_schema.key_column_usage kcu
    ON tc.constraint_name = kcu.constraint_name
   AND tc.table_schema   = kcu.table_schema
  WHERE tc.constraint_type = 'PRIMARY KEY'
    AND tc.table_schema = :'schema' AND tc.table_name = :'table'
  GROUP BY tc.table_schema, tc.table_name
),
fk AS (
  SELECT tc.table_schema, tc.table_name,
         jsonb_agg(jsonb_build_object(
           'constraint', tc.constraint_name,
           'columns', (
             SELECT jsonb_agg(k2.column_name ORDER BY k2.position_in_unique_constraint)
             FROM information_schema.key_column_usage k2
             WHERE k2.constraint_name = tc.constraint_name
               AND k2.table_schema    = tc.table_schema
           ),
           'ref_schema', ccu.table_schema,
           'ref_table',  ccu.table_name
         )) AS foreign_keys
  FROM information_schema.table_constraints tc
  JOIN information_schema.constraint_column_usage ccu
    ON tc.constraint_name = ccu.constraint_name
   AND tc.table_schema    = ccu.table_schema
  WHERE tc.constraint_type = 'FOREIGN KEY'
    AND tc.table_schema = :'schema' AND tc.table_name = :'table'
  GROUP BY tc.table_schema, tc.table_name
),
ix AS (
  SELECT schemaname AS table_schema, tablename AS table_name,
         jsonb_agg(jsonb_build_object('name', indexname, 'def', indexdef) ORDER BY indexname) AS indexes
  FROM pg_indexes
  WHERE schemaname = :'schema' AND tablename = :'table'
  GROUP BY schemaname, tablename
),
rls AS (
  SELECT schemaname AS table_schema, tablename AS table_name,
         jsonb_agg(jsonb_build_object(
           'policy', policyname, 'cmd', cmd, 'role', roles, 'using', qual, 'check', with_check
         ) ORDER BY policyname) AS rls_policies
  FROM pg_policies
  WHERE schemaname = :'schema' AND tablename = :'table'
  GROUP BY schemaname, tablename
)
SELECT jsonb_build_object(
  'schema', :'schema',
  'table',  :'table',
  'columns', (SELECT columns FROM cols),
  'primary_key', (SELECT primary_key FROM pk),
  'foreign_keys', (SELECT foreign_keys FROM fk),
  'indexes', (SELECT indexes FROM ix),
  'rls_policies', (SELECT rls_policies FROM rls)
);
