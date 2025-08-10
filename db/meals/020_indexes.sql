DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE schemaname='meals' AND indexname='idx_meal_plan_date') THEN
    CREATE INDEX idx_meal_plan_date ON meals.meal_plan (date);
  END IF;
END $$;
