from __future__ import annotations
import io, os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Correct paths for your repo layout
app_path = os.path.join(ROOT, "frontend", "src", "App.vue")
util_path = os.path.join(ROOT, "frontend", "src", "utils", "expandShorthand.js")
api_path = os.path.join(ROOT, "frontend", "src", "api.js")


def read(p):
    with io.open(p, "r", encoding="utf-8") as f:
        return f.read()


def write(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


# 1) Create helper file: frontend/src/utils/expandShorthand.js
if not os.path.exists(util_path):
    write(
        util_path,
        """// src/utils/expandShorthand.js
export function expandShorthand(input) {
  if (!input || !/^s:\\s*/i.test(input.trim())) return input; // pass-through
  const lines = input.split(/\\r?\\n/);
  let S = "", G = "", Q = "";
  for (const ln of lines) {
    const t = ln.trim();
    if (/^s:/i.test(t)) S = t.slice(2).trim();
    else if (/^g:/i.test(t)) G = t.slice(2).trim();
    else if (/^q:/i.test(t)) Q = t.slice(2).trim();
  }
  if (!S || !G || !Q) return input; // not valid shorthand
  return `STATE:\\n${S}\\n\\nGOAL:\\n${G}\\n\\nQUESTION:\\n${Q}`;
}
""",
    )

# 2) Patch App.vue
app_src = read(app_path)

# 2a) Add import
if "from './utils/expandShorthand'" not in app_src:
    app_src = app_src.replace(
        "import { sendQuery } from './api'",
        "import { sendQuery } from './api'\nimport { expandShorthand } from './utils/expandShorthand'",
    )

# 2b) Add shorthandEnabled + preview in data()
app_src = re.sub(
    r"(demoMode:\s*loadDemo\(\)\s*[\r\n]\s*})",
    r"demoMode: loadDemo(),\n      shorthandEnabled: false,\n      preview: null\n    }",
    app_src,
    count=1,
)

# 2c) Shorthand toggle after </form>
if "Shorthand (S/G/Q)" not in app_src:
    app_src = app_src.replace(
        "</form>",
        '</form>\n\n    <div class="row" style="display:flex;gap:16px;align-items:center;">\n      <label class="ck">\n        <input type="checkbox" v-model="shorthandEnabled" />\n        Shorthand (S/G/Q)\n      </label>\n    </div>',
    )

# 2d) Preview before error section
if "Expanded prompt preview" not in app_src:
    app_src = app_src.replace(
        '    <section v-if="error" class="error">{{ error }}</section>',
        '    <details v-if="preview" style="margin: 6px 0 10px;">\n      <summary style="cursor:pointer">Expanded prompt preview</summary>\n      <pre style="white-space:pre-wrap;background:#f6f6f8;padding:8px;border-radius:6px;margin-top:6px">{{ preview }}</pre>\n    </details>\n\n    <section v-if="error" class="error">{{ error }}</section>',
    )

# 2e) Expand prompt in onSubmit
app_src = re.sub(
    r"this\.loading = true\s*\n\s*const json = await sendQuery\(q\)",
    "this.loading = true\n        const expanded = this.shorthandEnabled ? expandShorthand(q) : q\n        this.preview = expanded !== q ? expanded : null\n\n        const json = await sendQuery(expanded, { prompt_director: this.shorthandEnabled })",
    app_src,
    count=1,
)

write(app_path, app_src)

# 3) Patch api.js to accept meta and send flag
api_src = read(api_path)
if "export async function sendQuery(text, meta" not in api_src:
    api_src = api_src.replace(
        "export async function sendQuery(text) {",
        "export async function sendQuery(text, meta = {}) {",
    )
    api_src = api_src.replace(
        "fetch('/api/route', {",
        "fetch((import.meta?.env?.VITE_BACKEND_URL) || '/api/route', {",
    )
    api_src = re.sub(
        r"body:\s*JSON\.stringify\(\{\s*text\s*\}\)",
        "body: JSON.stringify({ text, client_meta: { prompt_director: !!meta.prompt_director } })",
        api_src,
        count=1,
    )
    write(api_path, api_src)

print("[ok] Frontend shorthand toggle + preview installed.")
