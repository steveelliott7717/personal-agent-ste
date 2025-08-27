# Personal Agent (Browser + HTTP Adapter)

This repository provides a local **personal agent API** that can:
- Run scripted browser automation flows (via Playwright).
- Execute HTTP requests with guardrails (host allowlist, per-host rate limiting).
- Capture screenshots, PDFs, evaluations, and extracted content.
- Upload artifacts to destinations (filesystem, GitHub, Supabase, etc.).
- Support stealth browsing, proxies, geolocation, and timezone overrides.

---

## Features
- **Browser automation**: navigate, click, fill, type, wait, evaluate JS, extract content.
- **File upload**: send files to `<input type=file>` elements.
- **Artifacts**: screenshots and PDFs automatically tracked.
- **Destination upload**: copy artifacts to filesystem (and extendable to GitHub/Supabase).
- **Guardrails**: host allowlist, disallowed IP checks, and per-host rate limiting.
- **Stealth tweaks**: hide `navigator.webdriver`, spoof `maxTouchPoints`, set custom UA, locale, and timezone.
- **Persistent state**: save/load storage state (cookies + localStorage) between runs.

---

## Installation

```bash
git clone <this-repo-url>
cd personal-agent

# Install Python dependencies
pip install -r requirements.txt

# Install Playwright + browsers
pip install playwright
playwright install
```

Start the agent API (FastAPI/uvicorn assumed):

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Usage

The API exposes a single endpoint:

```
POST /app/api/agents/verb
```

Example verbs:
- `http.fetch`
- `browser.run`

---

### Example: Basic page extract
```bash
curl -sS -X POST http://localhost:8000/app/api/agents/verb   -H "Content-Type: application/json"   -d '{"verb":"browser.run","args":{"url":"https://example.com/","steps":[{"wait_for":{"selector":"h1"}},{"extract":{"selector":"h1","inner_text":true}}]}}'
```

---

### Example: Screenshot + artifact tracking
```bash
curl -sS -X POST http://localhost:8000/app/api/agents/verb   -H "Content-Type: application/json"   -d '{"verb":"browser.run","args":{"url":"https://example.com/","timeout_ms":15000,"steps":[{"wait_for":{"selector":"h1"}},{"screenshot":{"path":"example_artifact.png","full_page":true}}],"return_html":false}}'
```

---

### Example: Login to SauceDemo
```bash
curl -sS -X POST http://localhost:8000/app/api/agents/verb   -H "Content-Type: application/json"   -d '{"verb":"browser.run","args":{"url":"https://www.saucedemo.com/","timeout_ms":20000,"steps":[{"wait_for":{"selector":"#user-name","state":"visible"}},{"fill":{"selector":"#user-name","value":"standard_user"}},{"fill":{"selector":"#password","value":"secret_sauce"}},{"click":"#login-button"},{"wait_for":{"selector":".inventory_list","state":"visible"}},{"screenshot":{"path":"saucedemo.png","full_page":true}}]}}'
```

---

### Example: Destination upload (filesystem)
```bash
curl -sS -X POST http://localhost:8000/app/api/agents/verb   -H "Content-Type: application/json"   -d '{"verb":"browser.run","args":{"url":"https://httpbin.org/html","steps":[{"wait_for":{"selector":"h1"}},{"pdf":{"path":"verify.pdf"}},{"screenshot":{"path":"verify.png","full_page":true}}],"destination":{"type":"filesystem","dir":"downloads/artifacts"}}}'
```

Artifacts will be copied into `downloads/artifacts`.

---

## File Upload Example
```bash
curl -sS -X POST http://localhost:8000/app/api/agents/verb   -H "Content-Type: application/json"   -d '{"verb":"browser.run","args":{"url":"https://the-internet.herokuapp.com/upload","timeout_ms":30000,"steps":[{"wait_for":{"selector":"input#file-upload","state":"visible"}},{"set_files":{"selector":"input#file-upload","files":["C:\\Users\\Owner\\personal-agent\\README.md"]}},{"click":"#file-submit"},{"wait_for":{"selector":"#uploaded-files","state":"visible"}},{"screenshot":{"path":"upload_done.png","full_page":true}}]}}'
```

---

## Development Notes
- Modify `browser_adapter.py` for context creation, stealth settings, host guardrails, artifact tracking, and destination uploads.
- Payload JSONs can be saved under `payload/` and passed with `--data-binary @payload/file.json` to avoid escaping issues in curl.
- Extend `_upload_to_github`, `_upload_to_supabase_storage`, etc., if you need real cloud destinations.

---

## Roadmap
- [ ] Add real Supabase/GitHub backends for artifact upload.
- [ ] Add retry logic for flaky selectors.
- [ ] Add richer stealth (navigator.plugins, languages, etc.).
- [ ] CLI wrapper around API calls.
