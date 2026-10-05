from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_CHARS = 10_000
REQUIRED = ("learning_objectives", "prerequisites", "estimated_hours", "last_verified")


def body(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        return parts[2] if len(parts) == 3 else text
    return text


def main() -> int:
    failures: list[str] = []
    chapters = sorted((ROOT / "docs" / "chapters").glob("*.md"))
    for path in chapters:
        if path.name.lower() == 'readme.md':
            continue
        text = path.read_text(encoding="utf-8")
        header = text.split("---", 2)[1] if text.startswith("---") and text.count("---") >= 2 else ""
        missing = [key for key in REQUIRED if not re.search(rf"^{key}:", header, re.MULTILINE)]
        if missing:
            failures.append(f"{path}: missing front matter {', '.join(missing)}")
        count = len(body(text))
        if "draft: true" not in header and count < MIN_CHARS:
            failures.append(f"{path}: {count} characters, need at least {MIN_CHARS}")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"validated {len(chapters)} chapter(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
