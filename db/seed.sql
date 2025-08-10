-- Meals agent registry row (re-runnable)
INSERT INTO public.agents (slug, title, description, module_path, callable_name, namespaces, capabilities, status, version)
VALUES (
  'meals',
  'Meals',
  'Plan daily meals, swap items, and log completions.',
  'backend.agents.meals_agent',
  'class:MealsAgent',
  '["meals"]'::jsonb,
  '["Daylist","Swap","Complete","Plan"]'::jsonb,
  'enabled',
  'v1'
)
ON CONFLICT (slug) DO UPDATE
SET title=EXCLUDED.title,
    description=EXCLUDED.description,
    module_path=EXCLUDED.module_path,
    callable_name=EXCLUDED.callable_name,
    namespaces=EXCLUDED.namespaces,
    capabilities=EXCLUDED.capabilities,
    status=EXCLUDED.status,
    version=EXCLUDED.version;

-- Minimal instruction; you can add more tags later (planning/logging)
INSERT INTO public.agent_instructions (agent_name, tag, instructions)
VALUES (
  'meals', 'core',
  'You are a meals operator. Plan floating daily meal cards (no fixed times), allow swaps, and log completions. Read/write meals.recipe_templates, meals.meal_plan, meals.meal_log. Return ONLY JSON {thoughts, operations, response_template?}.'
);

-- Seed one recipe & a planned meal for today (safe re-run)
WITH r AS (
  INSERT INTO meals.recipe_templates (name, macros)
  VALUES ('Grilled Chicken + Rice + Broccoli', '{"cal":650,"protein":55,"fat":15,"carbs":75}'::jsonb)
  ON CONFLICT DO NOTHING
  RETURNING id
),
rid AS (
  SELECT id FROM r
  UNION ALL
  SELECT id FROM meals.recipe_templates WHERE name='Grilled Chicken + Rice + Broccoli' LIMIT 1
)
INSERT INTO meals.meal_plan (date, recipe_id, servings, status, freshness_rank, anchor)
SELECT CURRENT_DATE, rid.id, 1.0, 'planned', 0, NULL FROM rid
ON CONFLICT DO NOTHING;
