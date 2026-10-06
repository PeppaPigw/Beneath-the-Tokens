#!/usr/bin/env python3
"""Deterministic, standard-library content audit for the phase-two textbook.

The default mode is deliberately offline: URLs are listed as skipped evidence so
CI remains reproducible.  Pass ``--check-urls`` for an explicitly bounded HTTP
probe; HTTP 404s are failures while timeouts/network errors are reported
separately and do not become false 404s.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import socket
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RULES = SCRIPT_DIR / "audit_phase2_rules.json"


@dataclass
class Document:
    path: Path
    relative: str
    text: str
    frontmatter: dict[str, Any]
    headings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)


def _strip_comment(value: str) -> str:
    # Frontmatter values in this repository use comments only after whitespace.
    value = value.strip()
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def _scalar(value: str) -> Any:
    value = _strip_comment(value)
    if not value:
        return None
    if (value.startswith("\"") and value.endswith("\"")) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    low = value.lower()
    if low in {"null", "~"}:
        return None
    if low in {"true", "false"}:
        return low == "true"
    if re.fullmatch(r"[-+]?\d+", value):
        try:
            return int(value)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", value):
        try:
            return float(value)
        except ValueError:
            pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        # Commas in quoted labels are uncommon; retain a small quote-aware split.
        bits: list[str] = []
        current: list[str] = []
        quote: str | None = None
        for char in inner:
            if char in "'\"":
                if quote == char:
                    quote = None
                elif quote is None:
                    quote = char
            if char == "," and quote is None:
                bits.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        bits.append("".join(current).strip())
        return [_scalar(bit) for bit in bits if bit]
    return value


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by Markdown front matter.

    This intentionally avoids a YAML dependency. Unknown nested mappings are
    retained as strings; scalar fields and indented ``- item`` lists are parsed.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}
    result: dict[str, Any] = {}
    i = 1
    while i < end:
        line = lines[i]
        match = re.match(r"^([A-Za-z_][\w-]*):(?:\s*(.*))?$", line)
        if not match:
            i += 1
            continue
        key, raw = match.group(1), (match.group(2) or "")
        if raw.strip():
            result[key] = _scalar(raw)
            i += 1
            continue
        values: list[Any] = []
        j = i + 1
        while j < end:
            item = re.match(r"^\s+-\s*(.*)$", lines[j])
            if item:
                values.append(_scalar(item.group(1)))
                j += 1
                continue
            if lines[j].strip() and not lines[j].startswith((" ", "\t")):
                break
            j += 1
        result[key] = values if values else None
        i = j
    return result


def split_frontmatter(text: str) -> tuple[str, str]:
    """Return (front-matter text, body), tolerating missing front matter."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return "", text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "".join(lines[1:i]), "".join(lines[i + 1 :])
    return "", text


def extract_headings(text: str) -> list[str]:
    """Extract Markdown headings outside both backtick and tilde fences."""
    _, body = split_frontmatter(text)
    headings: list[str] = []
    active: tuple[str, int] | None = None
    for line in body.splitlines():
        fence = re.match(r"^\s*(`{3,}|~{3,})(?:\s*[^`]*)?$", line)
        if fence:
            marker, width = fence.group(1)[0], len(fence.group(1))
            if active is None:
                active = (marker, width)
            elif active[0] == marker and width >= active[1]:
                active = None
            continue
        if active is not None:
            continue
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
        if heading:
            headings.append(heading.group(1).strip().rstrip("#").strip())
    return headings


def check_fences(text: str) -> list[dict[str, Any]]:
    """Find unpaired Markdown backtick/tilde fences with line numbers."""
    errors: list[dict[str, Any]] = []
    active: tuple[str, int, int] | None = None
    for number, line in enumerate(text.splitlines(), 1):
        match = re.match(r"^\s*(`{3,}|~{3,})(?:\s*[^`]*)?$", line)
        if not match:
            continue
        marker = match.group(1)[0]
        width = len(match.group(1))
        if active is None:
            active = (marker, width, number)
        elif active[0] == marker and width >= active[1]:
            active = None
    if active is not None:
        errors.append({"class": "unpaired_fence", "line": active[2], "message": f"unclosed {active[0] * active[1]} fence"})
    return errors


def extract_urls(text: str) -> list[str]:
    # Strip angle punctuation while retaining query strings and fragments.
    found = re.findall(r"https?://[^\s)\\\]>\"']+", text)
    return sorted(set(url.rstrip(".,;:") for url in found))


def check_url(url: str, timeout: float = 5.0) -> dict[str, Any]:
    """Probe one URL and classify HTTP errors independently from timeouts."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "btt-phase2-audit/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - explicit audit URL
            status = int(response.getcode() or 200)
            return {"url": url, "status": status, "class": None if status < 400 else ("url_404" if status == 404 else "url_http_error")}
    except urllib.error.HTTPError as exc:
        # A number of small documentation servers reject HEAD while serving
        # GET normally. Retry those method errors once; preserve real 404s.
        if exc.code in {405, 501}:
            try:
                get_request = urllib.request.Request(url, method="GET", headers={"User-Agent": "btt-phase2-audit/1"})
                with urllib.request.urlopen(get_request, timeout=timeout) as response:  # noqa: S310
                    status = int(response.getcode() or 200)
                    return {"url": url, "status": status, "class": None if status < 400 else ("url_404" if status == 404 else "url_http_error")}
            except urllib.error.HTTPError as retry:
                exc = retry
            except (TimeoutError, socket.timeout) as retry:
                return {"url": url, "status": None, "class": "url_timeout", "message": str(retry) or "request timed out"}
            except urllib.error.URLError as retry:
                return {"url": url, "status": None, "class": "url_network_error", "message": str(retry)}
            except OSError as retry:
                return {"url": url, "status": None, "class": "url_network_error", "message": str(retry)}
        return {"url": url, "status": int(exc.code), "class": "url_404" if exc.code == 404 else "url_http_error", "message": str(exc)}
    except (TimeoutError, socket.timeout) as exc:
        return {"url": url, "status": None, "class": "url_timeout", "message": str(exc) or "request timed out"}
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower():
            return {"url": url, "status": None, "class": "url_timeout", "message": str(exc)}
        return {"url": url, "status": None, "class": "url_network_error", "message": str(exc)}
    except OSError as exc:
        return {"url": url, "status": None, "class": "url_network_error", "message": str(exc)}


def _error(kind: str, path: str, message: str, *, line: int | None = None, fatal: bool = True, **extra: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"class": kind, "path": path, "message": message, "fatal": fatal}
    if line is not None:
        item["line"] = line
    item.update(extra)
    return item


def _load_rules(path: Path | None) -> dict[str, Any]:
    target = path or DEFAULT_RULES
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read rules file {target}: {exc}") from exc


def _is_excluded(relative: str, rules: Mapping[str, Any]) -> bool:
    normalized = relative.replace("\\", "/")
    for pattern in rules.get("excluded_paths", []):
        pattern = str(pattern).replace("\\", "/")
        if normalized == pattern or normalized.startswith(pattern.rstrip("/") + "/") or normalized.endswith("/" + pattern):
            return True
    return False


def _chapter(relative: str, rules: Mapping[str, Any]) -> bool:
    # Preserve the historical single-pattern override used by callers and
    # fixtures; the default value delegates to the multi-pattern list.
    single = rules.get("chapter_glob")
    patterns = None if single in (None, "chapters/*.md") else [single]
    if patterns is None:
        patterns = rules.get("chapter_globs")
    if patterns is None:
        patterns = [rules.get("chapter_glob", "chapters/*.md")]
    elif isinstance(patterns, str):
        patterns = [patterns]
    return (
        any(fnmatch.fnmatch(relative, str(pattern)) for pattern in patterns)
        and relative.endswith(".md")
        and not _is_excluded(relative, rules)
    )


def _iter_documents(root: Path, rules: Mapping[str, Any]) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        if _is_excluded(rel, rules):
            continue
        text = path.read_text(encoding="utf-8")
        docs.append(Document(path, rel, text, parse_frontmatter(text), extract_headings(text)))
    return docs


def _sidebar_references(sidebar: Path) -> list[str]:
    """Read document strings from a Docusaurus sidebar without treating labels as paths."""
    if not sidebar or not sidebar.exists():
        return []
    text = sidebar.read_text(encoding="utf-8")
    refs: list[str] = []
    for match in re.finditer(r"['\"]([^'\"]+)['\"]", text):
        ref = match.group(1)
        before = text[max(0, match.start() - 80) : match.start()]
        after = text[match.end() : match.end() + 20]
        # Object keys and metadata values are not sidebar document refs.
        object_key = re.search(r"([A-Za-z_$][\w$]*)\s*:\s*$", before)
        if object_key:
            key = object_key.group(1)
            if key == "id":
                # Docusaurus also accepts object-form entries such as
                # {type: 'doc', id: 'chapters/intro'}.
                object_start = text.rfind("{", 0, match.start())
                object_end = text.rfind("}", 0, match.start())
                object_prefix = text[object_start : match.start()] if object_start > object_end else before
                if not re.search(r"\btype\s*:\s*['\"]doc['\"]", object_prefix):
                    continue
            elif key != "items":
                continue
        if re.match(r"\s*:", after):
            continue
        if ref in {"@docusaurus/plugin-content-docs"} or ref.startswith("@"):
            continue
        refs.append(ref)
    return list(dict.fromkeys(refs))


def _resolve_sidebar(ref: str, root: Path, by_rel: Mapping[str, Document], ids: Mapping[str, list[Document]]) -> Document | None:
    normalized = ref.removeprefix("./").removesuffix(".md")
    for candidate in (normalized + ".md", normalized + "/index.md"):
        if candidate in by_rel:
            return by_rel[candidate]
        # Excluded but real docs (for example chapter-template.md) still
        # legitimately appear in a sidebar and should not be called dangling.
        if (root / candidate).is_file():
            return Document(root / candidate, candidate, "", {}, [])
    final = normalized.rsplit("/", 1)[-1]
    if final in ids and len(ids[final]) == 1:
        return ids[final][0]
    return None


def _check_script(path: str, source: str, line_offset: int = 0) -> dict[str, Any] | None:
    try:
        ast.parse(source, filename=path)
    except SyntaxError as exc:
        return _error("script_syntax", path, exc.msg, line=(exc.lineno or 1) + line_offset, column=exc.offset)
    return None


def audit(docs_dir: str | Path, *, report_path: str | Path | None = None, sidebar_path: str | Path | None = None, rules_path: str | Path | None = None, check_urls: bool = False, url_timeout: float | None = None, max_url_workers: int | None = None) -> dict[str, Any]:
    root = Path(docs_dir).resolve()
    rules = _load_rules(Path(rules_path).resolve() if rules_path else None)
    docs = _iter_documents(root, rules)
    by_rel = {doc.relative: doc for doc in docs}
    ids: dict[str, list[Document]] = {}
    for doc in docs:
        value = doc.frontmatter.get("id")
        if isinstance(value, str) and value.strip():
            ids.setdefault(value.strip(), []).append(doc)

    all_errors: list[dict[str, Any]] = []
    for doc in docs:
        is_chapter = _chapter(doc.relative, rules)
        if is_chapter:
            for key in rules.get("required_frontmatter", []):
                if key not in doc.frontmatter or doc.frontmatter[key] is None:
                    doc.errors.append(_error("missing_frontmatter", doc.relative, f"missing frontmatter field: {key}", field=key))
            groups = rules.get("required_section_groups", {})
            lowered = "\n".join(doc.headings).lower()
            for name, terms in groups.items():
                if not any(str(term).lower() in lowered for term in terms):
                    doc.errors.append(_error("missing_pedagogical_section", doc.relative, f"missing pedagogical section group: {name}", section=name))
        for item in check_fences(doc.text):
            doc.errors.append(_error(item["class"], doc.relative, item["message"], line=item.get("line")))
        if not _is_excluded(doc.relative, rules):
            for pattern in rules.get("placeholder_patterns", []):
                match = re.search(pattern, doc.text, flags=re.IGNORECASE)
                if match:
                    line = doc.text[: match.start()].count("\n") + 1
                    doc.errors.append(_error("placeholder_word", doc.relative, f"placeholder pattern matched: {match.group(0)}", line=line, pattern=pattern))
                    break
        doc.checks.update({
            "frontmatter": bool(doc.frontmatter),
            "headings": len(doc.headings),
            "fences": not any(e["class"] == "unpaired_fence" for e in doc.errors),
            "placeholders": not any(e["class"] == "placeholder_word" for e in doc.errors),
            "pedagogical_sections": not any(e["class"] == "missing_pedagogical_section" for e in doc.errors),
            "prerequisites": True,
            "script_syntax": True,
            "urls": len(extract_urls(doc.text)),
        })
        all_errors.extend(doc.errors)

    for value, owners in ids.items():
        if len(owners) > 1:
            for owner in owners:
                err = _error("duplicate_id", owner.relative, f"duplicate frontmatter id: {value}", id=value, files=[x.relative for x in owners])
                owner.errors.append(err)
                all_errors.append(err)

    # Prerequisites must name an existing frontmatter id.
    for doc in docs:
        prereqs = doc.frontmatter.get("prerequisites")
        if not isinstance(prereqs, list):
            continue
        for prereq in prereqs:
            if isinstance(prereq, str) and prereq and prereq not in ids:
                err = _error("dangling_prerequisite", doc.relative, f"prerequisite id does not exist: {prereq}", id=prereq)
                doc.errors.append(err)
                all_errors.append(err)

    if sidebar_path:
        sidebar = Path(sidebar_path).resolve()
    else:
        # Prefer the production Docusaurus config, then make isolated fixture
        # roots convenient by accepting docs/sidebars.ts or a sibling config.
        candidates = (root.parent / "website" / "sidebars.ts", root / "sidebars.ts", root.parent / "sidebars.ts")
        sidebar = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    sidebar_errors: list[dict[str, Any]] = []
    for ref in _sidebar_references(sidebar):
        if _resolve_sidebar(ref, root, by_rel, ids) is None:
            sidebar_errors.append(_error("dangling_sidebar", str(sidebar), f"sidebar reference does not resolve: {ref}", id=ref))
    all_errors.extend(sidebar_errors)

    # Check Python snippets and repository scripts without executing untrusted code.
    for doc in docs:
        lines = doc.text.splitlines()
        active_fence: tuple[str, int] | None = None
        language = ""
        start = 0
        content: list[str] = []
        languages = set(rules.get("script_languages", ["python", "py", "python3"]))
        for number, line in enumerate(lines, 1):
            fence = re.match(r"^\s*(`{3,}|~{3,})(?:\s*([\w+-]+))?(?:\s+.*)?\s*$", line)
            if fence and active_fence is None:
                active_fence = (fence.group(1)[0], len(fence.group(1)))
                language, start, content = (fence.group(2) or "").lower(), number, []
                continue
            if fence and active_fence is not None and fence.group(1)[0] == active_fence[0] and len(fence.group(1)) >= active_fence[1]:
                if language in languages:
                    syntax_error = _check_script(f"{doc.relative}:{start}", "\n".join(content), start)
                    if syntax_error:
                        doc.errors.append(syntax_error)
                        all_errors.append(syntax_error)
                active_fence = None
                continue
            if active_fence is not None:
                content.append(line)
    scripts_root = root.parent / "scripts"
    if scripts_root.exists():
        for path in sorted(scripts_root.rglob("*.py")):
            try:
                syntax_error = _check_script(path.relative_to(root.parent).as_posix(), path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError) as exc:
                syntax_error = _error("script_syntax", path.relative_to(root.parent).as_posix(), str(exc))
            if syntax_error:
                all_errors.append(syntax_error)

    urls: dict[str, list[str]] = {}
    for doc in docs:
        for url in extract_urls(doc.text):
            urls.setdefault(url, []).append(doc.relative)
    network_checks: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if check_urls and urls:
        timeout = float(url_timeout if url_timeout is not None else rules.get("url_timeout_seconds", 5))
        workers = max(1, min(int(max_url_workers if max_url_workers is not None else rules.get("max_url_workers", 8)), 8, len(urls)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(check_url, url, timeout): url for url in urls}
            for future in as_completed(future_map):
                result = future.result()
                result["paths"] = urls[result["url"]]
                network_checks.append(result)
                if result.get("class") in {"url_404", "url_http_error"}:
                    err = _error(result["class"], ",".join(result["paths"]), f"URL check failed: {result['url']}", fatal=True, url=result["url"], status=result.get("status"))
                    all_errors.append(err)
                elif result.get("class"):
                    # Network conditions are intentionally non-fatal in CI, but
                    # still belong in the report's error classes for triage.
                    result["fatal"] = False
                    all_errors.append(_error(result["class"], ",".join(result["paths"]), f"URL check {result['class']}: {result['url']}", fatal=False, url=result["url"], status=result.get("status")))
        # Thread completion order is nondeterministic; reports must be stable.
        network_checks.sort(key=lambda item: str(item.get("url", "")))
    else:
        skipped = [{"url": url, "paths": paths, "reason": "network_check_disabled"} for url, paths in sorted(urls.items())]

    for doc in docs:
        # Errors added after the first aggregation (duplicate/prerequisite/syntax)
        # are reflected in the per-file checks as well as the error list.
        classes = {str(error.get("class")) for error in doc.errors}
        doc.checks["prerequisites"] = "dangling_prerequisite" not in classes
        doc.checks["script_syntax"] = "script_syntax" not in classes
        doc.checks["error_count"] = len(doc.errors)
    fatal_errors = [e for e in all_errors if e.get("fatal", True)]
    error_classes = sorted({str(e.get("class")) for e in all_errors})
    report: dict[str, Any] = {
        "version": 1,
        "docs": str(root),
        "passed": not fatal_errors,
        "structural_passed": not any(e.get("fatal", True) for e in all_errors if not str(e.get("class", "")).startswith("url_")),
        "summary": {"files": len(docs), "errors": len(all_errors), "fatal_errors": len(fatal_errors), "error_classes": error_classes, "urls": len(urls)},
        "error_classes": error_classes,
        "errors": all_errors,
        "files": [{"path": doc.relative, "id": doc.frontmatter.get("id"), "checks": doc.checks, "errors": doc.errors} for doc in docs],
        "network_checks": network_checks,
        "skipped_network_checks": skipped,
    }
    if report_path:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="docs", help="Markdown documentation root")
    parser.add_argument("--report", default="reports/phase2-content.json", help="JSON report destination")
    parser.add_argument("--sidebar", help="optional sidebars.ts path")
    parser.add_argument("--rules", help="optional rules JSON path")
    parser.add_argument("--check-urls", action="store_true", help="enable bounded HTTP URL checks")
    parser.add_argument("--url-timeout", type=float, help="per-request timeout in seconds")
    parser.add_argument("--max-url-workers", type=int, help="maximum concurrent URL checks (capped at 8)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = audit(args.docs, report_path=args.report, sidebar_path=args.sidebar, rules_path=args.rules, check_urls=args.check_urls, url_timeout=args.url_timeout, max_url_workers=args.max_url_workers)
    except ValueError as exc:
        print(f"audit configuration error: {exc}", file=sys.stderr)
        return 2
    for error in result["errors"]:
        print(f"{error['class']}: {error['path']}: {error['message']}", file=sys.stderr)
    print(f"phase2 audit: {'PASS' if result['passed'] else 'FAIL'} ({result['summary']['errors']} errors; report {args.report})")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
