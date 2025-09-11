# Anchor Rules Engine

The Anchor Rules Engine provides a safe, idempotent way to apply additive or replacement edits to files without needing to generate a full file replacement. This is particularly useful when an LLM returns a small code snippet instead of a complete file, or when you want to guarantee that a change is applied in a minimal, non-destructive way.

The core of the engine is in `backend/repo/anchors.py`.

## Key Concepts

### The `Anchor` Dataclass

An `Anchor` defines a rule for an edit. It has the following fields:

- `name` (str): A unique name for the anchor (e.g., `fastapi_middleware`).
- `file_glob` (str): A glob pattern to match against file paths (e.g., `*.py`).
- `pattern` (str): A Python regex pattern to find the location in the file to apply the change.
- `insert_mode` (Literal): How to apply the payload relative to the `pattern`.
  - `before`: Insert the payload immediately before the matched text.
  - `after`: Insert the payload immediately after the matched text.
  - `replace`: Replace the matched text with the payload.
  - `append_end`: Append the payload to the end of the file (the pattern is ignored).
- `unique` (bool): If `True`, the engine will first check if the payload already exists in the file. If it does, the operation is skipped, ensuring idempotency. Defaults to `True`.

### Applying Anchors

The `apply_anchor` function takes the original file content, an `Anchor` object, and the payload string, and returns the modified content.

The `apply_many` function can apply a list of anchors sequentially.

## Defining New Anchors

To define a new anchor, simply add a new `Anchor` instance to the `BUILTIN_ANCHORS` list in `backend/repo/anchors.py`.

**Example: An anchor to add a new route to a FastAPI router.**

```python
Anchor(
    name="fastapi_add_route",
    file_glob="backend/api.py",
    # Find the last app.include_router(...) line
    pattern=r"^(app\.include_router\(.*\))\n",
    insert_mode="after",
    unique=True,
)
```

## CLI Tool

A command-line tool, `tools/anchor_apply.py`, is provided for testing anchors.

**Usage:**
```bash
python tools/anchor_apply.py --file path/to/your/file.py --anchor <anchor_name> --payload-file path/to/payload.txt
```

This will print the modified file content to standard output.
