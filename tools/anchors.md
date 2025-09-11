# Anchor Rules Engine

The anchor system applies safe, additive edits using regex anchors.

## Built-ins
- python_imports
- fastapi_middleware
- json_config_start / json_config_end
- ts_imports

## Programmatic use
from backend.registry.anchors import find_anchor, apply_anchor
a = find_anchor("fastapi_middleware")
new_text, changed = apply_anchor(text, a, "app.add_middleware(RequestLoggingMiddleware)")

## CLI
python tools/anchor_apply.py --file backend/api.py --anchor fastapi_middleware --payload "app.add_middleware(RequestLoggingMiddleware)"

Run twice to check idempotency.
