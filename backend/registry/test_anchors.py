# tests/test_anchors.py
import pytest
from backend.repo.anchors import Anchor, apply_anchor


@pytest.fixture
def sample_text():
    return """
import os
import sys

def main():
    print("hello")

if __name__ == "__main__":
    main()
"""


def test_insert_after(sample_text):
    anchor = Anchor(
        name="after_imports",
        file_glob="*.py",
        pattern=r"import sys\n",
        insert_mode="after",
    )
    payload = "import re"
    new_text, changed = apply_anchor(sample_text, anchor, payload)
    assert changed
    assert "import sys\nimport re" in new_text


def test_insert_before(sample_text):
    anchor = Anchor(
        name="before_main",
        file_glob="*.py",
        pattern=r"def main\(\):",
        insert_mode="before",
    )
    payload = "# New function here\ndef helper():\n    pass\n"
    new_text, changed = apply_anchor(sample_text, anchor, payload)
    assert changed
    assert "def helper():\n    pass\n\ndef main():" in new_text


def test_replace(sample_text):
    anchor = Anchor(
        name="replace_print",
        file_glob="*.py",
        pattern=r'print\("hello"\)',
        insert_mode="replace",
    )
    payload = 'print("world")'
    new_text, changed = apply_anchor(sample_text, anchor, payload)
    assert changed
    assert 'print("world")' in new_text
    assert 'print("hello")' not in new_text


def test_append_end(sample_text):
    anchor = Anchor(
        name="append_end", file_glob="*.py", pattern="", insert_mode="append_end"
    )
    payload = "# Appended comment"
    new_text, changed = apply_anchor(sample_text, anchor, payload)
    assert changed
    assert new_text.strip().endswith("# Appended comment")


def test_idempotency_unique(sample_text):
    anchor = Anchor(
        name="after_imports",
        file_glob="*.py",
        pattern=r"import sys\n",
        insert_mode="after",
        unique=True,
    )
    payload = "import re"

    # First application
    text1, changed1 = apply_anchor(sample_text, anchor, payload)
    assert changed1
    assert "import sys\nimport re" in text1

    # Second application should do nothing
    text2, changed2 = apply_anchor(text1, anchor, payload)
    assert not changed2
    assert text1 == text2


def test_idempotency_not_unique(sample_text):
    anchor = Anchor(
        name="after_imports",
        file_glob="*.py",
        pattern=r"import sys\n",
        insert_mode="after",
        unique=False,
    )
    payload = "import re"

    text1, changed1 = apply_anchor(sample_text, anchor, payload)
    assert changed1

    # Second application should add it again
    text2, changed2 = apply_anchor(text1, anchor, payload)
    assert changed2
    assert text1 != text2
    assert text2.count("import re") == 2


def test_no_match(sample_text):
    anchor = Anchor(
        name="no_match",
        file_glob="*.py",
        pattern=r"non_existent_pattern",
        insert_mode="after",
    )
    payload = "this won't be inserted"
    new_text, changed = apply_anchor(sample_text, anchor, payload)
    assert not changed
    assert new_text == sample_text
