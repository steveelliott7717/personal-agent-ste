CREATE TABLE IF NOT EXISTS meals.recipe_templates (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name        text NOT NULL,
  macros      jsonb,                         -- {cal, protein, fat, carbs}
  meta        jsonb DEFAULT '{}'::jsonb,
  created_at  timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS meals.meal_plan (
  id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  date           date NOT NULL,
  recipe_id      uuid,
  servings       numeric(6,2) DEFAULT 1.0,
  status         text DEFAULT 'planned',   -- 'planned'|'done'|'skipped'
  freshness_rank int DEFAULT 0,
  anchor         text,                     -- 'breakfast'|'lunch'|'dinner'|'snack' (optional)
  calc_macros    jsonb,                    -- optional computed macros per plan
  created_at     timestamptz DEFAULT now(),

  CONSTRAINT fk_meal_plan_recipe
    FOREIGN KEY (recipe_id) REFERENCES meals.recipe_templates(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS meals.meal_log (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  meal_plan_id  uuid,
  ts            timestamptz DEFAULT now(),
  notes         text,
  created_at    timestamptz DEFAULT now(),

  CONSTRAINT fk_meal_log_plan
    FOREIGN KEY (meal_plan_id) REFERENCES meals.meal_plan(id) ON DELETE SET NULL
);
