param(
  [string]$Db = $env:SUPABASE_DB_URL
)

if (-not $Db) {
  Write-Error "Set SUPABASE_DB_URL env var or pass -Db"
  exit 1
}

$files = @(
  "db/public/000_extensions.sql",
  "db/public/010_registry.sql",
  "db/public/020_router.sql",
  "db/public/030_data_bus.sql",
  "db/public/040_embeddings.sql",
  "db/meals/000_schema.sql",
  "db/meals/010_tables.sql",
  "db/meals/020_indexes.sql",
  # "db/meals/030_policies.sql",
  "db/seed.sql"
)

foreach ($f in $files) {
  Write-Host "Applying $f"
  psql "$Db" -f $f
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Done."
