import os, requests

BASE = os.getenv("AGENTS_BASE", "http://localhost:8000")
URL = f"{BASE}/app/api/agents/verb"


def call(payload: dict):
    r = requests.post(URL, json=payload, timeout=45)
    r.raise_for_status()
    return r.json()


def test_browser_too_many_steps():
    # build > MAX_STEPS steps; default cap is 50
    steps = [{}] * 60
    out = call(
        {"verb": "browser.run", "args": {"url": "https://example.com/", "steps": steps}}
    )
    assert out["ok"] is False
    assert out["error"]["code"] in ("BadRequest", "AdapterError")
    assert "too many steps" in out["error"]["message"]


def test_browser_evaluate_missing_js():
    out = call(
        {
            "verb": "browser.run",
            "args": {
                "url": "https://example.com/",
                "steps": [{"evaluate": {"args": [1, 2, 3]}}],
            },
        }
    )
    assert out["ok"] is False
    assert out["error"]["code"] in ("BadRequest", "AdapterError")
    assert "evaluate must include js" in out["error"]["message"]


def test_browser_set_files_missing_local_file():
    out = call(
        {
            "verb": "browser.run",
            "args": {
                "url": "https://www.w3schools.com/howto/howto_html_file_upload_button.asp",
                "steps": [
                    {"wait_for": {"selector": "input[type=file]"}},
                    {
                        "set_files": {
                            "selector": "input[type=file]",
                            "files": ["does_not_exist_123.txt"],
                        }
                    },
                ],
            },
        }
    )
    assert out["ok"] is False
    assert out["error"]["code"] in ("BadRequest", "AdapterError")
    assert "missing local files" in out["error"]["message"]


def test_browser_happy_path_title_and_screenshot(tmp_path):
    shot = tmp_path / "ok.png"
    out = call(
        {
            "verb": "browser.run",
            "args": {
                "url": "https://example.com/",
                "steps": [
                    {"wait_for": {"selector": "body"}},
                    {"screenshot": {"path": str(shot)}},
                ],
                "return_html": False,
            },
        }
    )
    assert out["ok"] is True
    res = out["result"]
    assert res["status"] == 200
    assert "Example Domain" in (res.get("title") or "")
