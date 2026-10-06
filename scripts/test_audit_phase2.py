from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

import audit_phase2


SECTIONS = """\
## Why this problem exists
## One-sentence mental model
## System mechanism
## Measured experiment
## Failure clinic
## Six comprehension checks
## Exercises
## Source map
"""


def chapter(identifier: str, *, prerequisites: str = "[]", body: str = SECTIONS) -> str:
    return f"""---
id: {identifier}
title: Test chapter
sidebar_position: 1
prerequisites: {prerequisites}
---
# Test chapter
{body}
"""


class AuditPhase2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "docs"
        (self.root / "chapters").mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def run_audit(self, *, sidebar: Path | None = None, check_urls: bool = False) -> dict:
        return audit_phase2.audit(self.root, sidebar_path=sidebar, check_urls=check_urls)

    def test_duplicate_ids_identifies_both_files(self) -> None:
        self.write("chapters/a.md", chapter("same"))
        self.write("chapters/b.md", chapter("same"))
        result = self.run_audit()
        errors = [e for e in result["errors"] if e["class"] == "duplicate_id"]
        self.assertEqual({e["path"] for e in errors}, {"chapters/a.md", "chapters/b.md"})
        self.assertFalse(result["passed"])

    def test_dangling_prerequisite_and_sidebar(self) -> None:
        self.write("chapters/a.md", chapter("a", prerequisites="[missing]"))
        sidebar = Path(self.temp.name) / "sidebars.ts"
        sidebar.write_text("export default {book: ['chapters/a', 'chapters/nope']};", encoding="utf-8")
        result = self.run_audit(sidebar=sidebar)
        classes = {e["class"] for e in result["errors"]}
        self.assertIn("dangling_prerequisite", classes)
        self.assertIn("dangling_sidebar", classes)

    def test_missing_pedagogical_section(self) -> None:
        self.write("chapters/a.md", chapter("a", body="## Why this problem exists\n"))
        result = self.run_audit()
        self.assertTrue(any(e["class"] == "missing_pedagogical_section" for e in result["errors"]))

    def test_headings_inside_backtick_and_tilde_fences_are_ignored(self) -> None:
        body = """```markdown
## Why this problem exists
## System mechanism
```
~~~markdown
## Measured experiment
## Failure clinic
~~~
"""
        headings = audit_phase2.extract_headings(body)
        self.assertNotIn("Why this problem exists", headings)
        self.assertNotIn("Measured experiment", headings)

    def test_tilde_python_fence_is_syntax_checked(self) -> None:
        body = SECTIONS + "\n~~~python\ndef broken(:\n  pass\n~~~\n"
        self.write("chapters/a.md", chapter("a", body=body))
        result = self.run_audit()
        self.assertTrue(any(e["class"] == "script_syntax" for e in result["errors"]))

    def test_custom_chapter_glob_from_rules_is_honored(self) -> None:
        self.write("units/a.md", chapter("a", body="## Why this problem exists\n"))
        rules = json.loads(audit_phase2.DEFAULT_RULES.read_text(encoding="utf-8"))
        rules["chapter_glob"] = "units/*.md"
        rules_path = Path(self.temp.name) / "rules.json"
        rules_path.write_text(json.dumps(rules), encoding="utf-8")
        result = audit_phase2.audit(self.root, rules_path=rules_path)
        self.assertTrue(any(e["class"] == "missing_pedagogical_section" and e["path"] == "units/a.md" for e in result["errors"]))

    def test_sidebar_labels_and_metadata_are_not_document_refs(self) -> None:
        sidebar = Path(self.temp.name) / "sidebars.ts"
        sidebar.write_text("""export default {bookSidebar: [{type: 'category', label: 'Overview', items: ['chapters/a'], custom: 'ignored'}]};""", encoding="utf-8")
        self.assertEqual(audit_phase2._sidebar_references(sidebar), ["chapters/a"])

    def test_sidebar_object_doc_id_is_checked(self) -> None:
        self.write("chapters/a.md", chapter("a"))
        sidebar = Path(self.temp.name) / "sidebars.ts"
        sidebar.write_text("export default {bookSidebar: [{type: 'doc', id: 'missing-doc'}]};", encoding="utf-8")
        result = self.run_audit(sidebar=sidebar)
        self.assertIn("missing-doc", [e.get("id") for e in result["errors"] if e["class"] == "dangling_sidebar"])


    def test_unpaired_fence_and_placeholder(self) -> None:
        self.write("chapters/a.md", chapter("a", body=SECTIONS + "\nTODO: fill this in\n```python\nprint('x')\n"))
        result = self.run_audit()
        classes = {e["class"] for e in result["errors"]}
        self.assertIn("unpaired_fence", classes)
        self.assertIn("placeholder_word", classes)

    def test_python_script_syntax(self) -> None:
        self.write("chapters/a.md", chapter("a", body=SECTIONS + "\n```python\ndef broken(:\n  pass\n```\n"))
        result = self.run_audit()
        self.assertTrue(any(e["class"] == "script_syntax" for e in result["errors"]))

    def test_python_fence_info_attributes_are_parsed(self) -> None:
        body = SECTIONS + "\n```python title=\"example.py\"\ndef broken(:\n  pass\n```\n"
        self.write("chapters/a.md", chapter("a", body=body))
        result = self.run_audit()
        self.assertTrue(any(e["class"] == "script_syntax" for e in result["errors"]))

    def test_repository_python_script_syntax(self) -> None:
        scripts = self.root.parent / "scripts"
        scripts.mkdir()
        (scripts / "broken.py").write_text("def broken(:\n", encoding="utf-8")
        self.write("chapters/a.md", chapter("a"))
        result = self.run_audit()
        self.assertTrue(any(e["class"] == "script_syntax" and e["path"] == "scripts/broken.py" for e in result["errors"]))

    def test_default_url_checks_are_skipped(self) -> None:
        self.write("chapters/a.md", chapter("a", body=SECTIONS + "\n[spec](https://example.invalid/spec)\n"))
        result = self.run_audit()
        self.assertEqual(len(result["network_checks"]), 0)
        self.assertEqual(result["skipped_network_checks"][0]["reason"], "network_check_disabled")

    @mock.patch("urllib.request.urlopen")
    def test_url_404_is_distinct(self, urlopen: mock.Mock) -> None:
        urlopen.side_effect = HTTPError("https://example.invalid/missing", 404, "missing", {}, None)
        self.write("chapters/a.md", chapter("a", body=SECTIONS + "\nhttps://example.invalid/missing\n"))
        result = self.run_audit(check_urls=True)
        self.assertTrue(any(e["class"] == "url_404" for e in result["errors"]))

    @mock.patch("urllib.request.urlopen")
    def test_url_timeout_is_not_404(self, urlopen: mock.Mock) -> None:
        urlopen.side_effect = socket.timeout("timed out")
        self.write("chapters/a.md", chapter("a", body=SECTIONS + "\nhttps://example.invalid/slow\n"))
        result = self.run_audit(check_urls=True)
        self.assertFalse(any(e["class"] == "url_404" for e in result["errors"]))
        self.assertEqual(result["network_checks"][0]["class"], "url_timeout")
        self.assertTrue(result["passed"])

    def test_network_report_order_is_deterministic(self) -> None:
        body = SECTIONS + "\nhttps://example.invalid/z\nhttps://example.invalid/a\n"
        self.write("chapters/a.md", chapter("a", body=body))
        with mock.patch.object(audit_phase2, "check_url", side_effect=lambda url, timeout: {"url": url, "status": 200, "class": None}):
            result = self.run_audit(check_urls=True)
        self.assertEqual([item["url"] for item in result["network_checks"]], ["https://example.invalid/a", "https://example.invalid/z"])

    def test_report_is_json_and_per_file(self) -> None:
        self.write("chapters/a.md", chapter("a"))
        output = Path(self.temp.name) / "reports" / "phase2.json"
        result = audit_phase2.audit(self.root, report_path=output)
        self.assertTrue(result["passed"])
        parsed = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(parsed["files"][0]["path"], "chapters/a.md")
        self.assertIn("error_classes", parsed)


if __name__ == "__main__":
    unittest.main()
