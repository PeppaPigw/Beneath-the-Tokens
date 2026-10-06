"""Deterministic validators for the phase-two chapter and evidence contracts.

This module deliberately performs contract checks only.  It does not score prose,
count characters, or claim that a list of headings proves that a chapter teaches
well.  Editorial review remains the place for those judgements.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Mapping

try:  # Works both as ``python scripts/...`` and as a package import.
    from audit_phase2 import parse_frontmatter, split_frontmatter
except ModuleNotFoundError:  # pragma: no cover - exercised by module invocation
    from scripts.audit_phase2 import parse_frontmatter, split_frontmatter


LEGACY_FIELDS = ("id", "title", "description", "slug", "sidebar_position")
PHASE2_FIELDS = (
    "phase",
    "chapter_number",
    "level",
    "prerequisites",
    "learning_objectives",
    "paper_count",
    "source_commits",
    "lab_paths",
    "last_verified",
)
LEVELS = {"foundation", "core", "systems", "advanced", "frontier", "capstone"}
EVIDENCE_TYPES = {
    "fact",
    "definition",
    "derivation",
    "paper_result",
    "source_observation",
    "experiment_measurement",
    "inference",
    "design_judgment",
    "unverified_hypothesis",
}
EVIDENCE_FIELDS = (
    "claim",
    "type",
    "source_url",
    "version",
    "experiment_id",
    "limitation",
    "review_date",
)
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_URL_RE = re.compile(r"^https?://[^\s]+$")
_BRANCH_NAMES = {"main", "master", "latest", "head", "develop", "dev"}


def _error(path: str, kind: str, message: str, **extra: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"class": kind, "path": path, "message": message}
    item.update(extra)
    return item


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))


def _is_relative_path(value: Any) -> bool:
    if not _is_nonempty_string(value):
        return False
    path = value.replace("\\", "/")
    return not path.startswith("/") and path != "." and ".." not in path.split("/")

# The opening contract is intentionally narrow: the first substantive teaching
# milestones are problem → mental model → mechanism.  An author may place a
# short example or a formal definition between the latter two.  We do not use
# heading counts as a proxy for depth; later sections are reviewed editorially.
_MILESTONE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("problem", ("问题边界", "why this problem exists", "problem boundary", "具体困惑")),
    ("mental_model", ("直觉模型", "mental model", "心智模型", "intuition")),
    ("mechanism", ("机制与源码入口", "system mechanism", "机制", "调用链")),
)
_SUPPORTING_ALIASES: tuple[str, ...] = (
    "最小例子",
    "minimal example",
    "micro example",
    "手算例子",
    "正式定义",
    "formal definition",
    "first-principles",
    "推导",
)
_INTRO_ALIASES: tuple[str, ...] = (
    "本章地图",
    "chapter map",
    "overview",
    "导读",
    "学习目标",
    "learning objectives",
    "前置",
    "prerequisite",
    "导航",
)


def _section_headings(text: str) -> list[str]:
    """Return h2–h6 headings outside fenced code and YAML frontmatter.

    The h1 title and prose before the first section are presentation, not a
    substantive teaching section, so they cannot accidentally satisfy the order
    contract.
    """
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
        heading = re.match(r"^\s{0,3}#{2,6}\s+(.+?)\s*$", line)
        if heading:
            headings.append(heading.group(1).strip().rstrip("#").strip())
    return headings


def _matches(value: str, aliases: tuple[str, ...]) -> bool:
    folded = value.casefold()
    return any(alias.casefold() in folded for alias in aliases)


def _beginner_sequence_errors(path: str, text: str) -> list[dict[str, Any]]:
    headings = _section_headings(text)
    expected = [name for name, _ in _MILESTONE_ALIASES]
    milestone_index = 0
    for heading in headings:
        if _matches(heading, _INTRO_ALIASES):
            continue
        matched = next((name for name, aliases in _MILESTONE_ALIASES if _matches(heading, aliases)), None)
        if matched is None and _matches(heading, _SUPPORTING_ALIASES):
            # A hand-sized example or a derivation can sit between intuition and
            # mechanism without becoming the next conceptual milestone.
            if milestone_index >= 1:
                continue
        if matched is None:
            return [
                _error(
                    path,
                    "beginner_sequence",
                    "the first substantive sections must follow problem → mental model → mechanism; unexpected heading before that sequence: "
                    + heading,
                    expected=expected,
                    heading=heading,
                )
            ]
        if expected[milestone_index] != matched:
            return [
                _error(
                    path,
                    "beginner_sequence",
                    "the first substantive sections must follow problem → mental model → mechanism (expected "
                    + expected[milestone_index]
                    + ", found "
                    + matched
                    + ")",
                    expected=expected,
                    heading=heading,
                )
            ]
        milestone_index += 1
        if milestone_index == len(expected):
            return []
    missing = expected[milestone_index:]
    return [
        _error(
            path,
            "beginner_sequence",
            "the first substantive sections must follow problem → mental model → mechanism (missing: "
            + ", ".join(missing)
            + ")",
            expected=expected,
            missing=missing,
        )
    ]

def validate_chapter_file(path: str, text: str) -> list[dict[str, Any]]:
    """Validate one phase-two Markdown chapter and return stable error records.

    Errors are emitted in contract order, then the single pedagogical-sequence
    observation.  This makes CI output reproducible across Python versions.
    """
    errors: list[dict[str, Any]] = []
    frontmatter = parse_frontmatter(text)
    if not frontmatter:
        return [_error(path, "missing_frontmatter", "chapter must start with a closed YAML frontmatter block")]

    missing_legacy = [field for field in LEGACY_FIELDS if field not in frontmatter]
    if missing_legacy:
        errors.append(
            _error(
                path,
                "missing_legacy_frontmatter",
                "一期-compatible fields are required: " + ", ".join(missing_legacy),
                fields=missing_legacy,
            )
        )

    missing_phase2 = [field for field in PHASE2_FIELDS if field not in frontmatter]
    if missing_phase2:
        errors.append(
            _error(
                path,
                "missing_frontmatter",
                "phase-two fields are required: " + ", ".join(missing_phase2),
                fields=missing_phase2,
            )
        )

    if "phase" in frontmatter and (not _is_int(frontmatter["phase"]) or frontmatter["phase"] != 2):
        errors.append(_error(path, "invalid_phase", "phase must be integer 2"))

    if "chapter_number" in frontmatter and (
        not _is_int(frontmatter["chapter_number"]) or not 1 <= frontmatter["chapter_number"] <= 40
    ):
        errors.append(_error(path, "invalid_chapter_number", "chapter_number must be an integer from 1 through 40"))

    if "level" in frontmatter and frontmatter["level"] not in LEVELS:
        errors.append(_error(path, "invalid_level", "level must be one of: " + ", ".join(sorted(LEVELS))))

    prerequisites = frontmatter.get("prerequisites")
    if "prerequisites" in frontmatter and (
        not isinstance(prerequisites, list) or any(not isinstance(item, str) or not _ID_RE.fullmatch(item) for item in prerequisites)
    ):
        errors.append(_error(path, "invalid_prerequisites", "prerequisites must be a list of stable chapter ids"))

    objectives = frontmatter.get("learning_objectives")
    if "learning_objectives" in frontmatter and (
        not isinstance(objectives, list) or not 2 <= len(objectives) <= 12 or any(not _is_nonempty_string(item) for item in objectives)
    ):
        errors.append(_error(path, "invalid_learning_objectives", "learning_objectives must contain 2–12 non-empty strings"))

    if "paper_count" in frontmatter and (not _is_int(frontmatter["paper_count"]) or frontmatter["paper_count"] < 0):
        errors.append(_error(path, "invalid_paper_count", "paper_count must be a non-negative integer"))

    commits = frontmatter.get("source_commits")
    if "source_commits" in frontmatter and (
        not isinstance(commits, list)
        or any(not _is_nonempty_string(item) or any(char.isspace() for char in item) or item.casefold() in _BRANCH_NAMES for item in commits)
    ):
        errors.append(_error(path, "invalid_source_commits", "source_commits must list pinned commit hashes or tags, never branch names"))

    labs = frontmatter.get("lab_paths")
    if "lab_paths" in frontmatter and (not isinstance(labs, list) or any(not _is_relative_path(item) for item in labs)):
        errors.append(_error(path, "invalid_lab_paths", "lab_paths must contain repository-relative paths without '..'"))

    if "last_verified" in frontmatter and not _is_iso_date(frontmatter["last_verified"]):
        errors.append(_error(path, "invalid_last_verified", "last_verified must be an ISO date (YYYY-MM-DD)"))

    errors.extend(_beginner_sequence_errors(path, text))
    return errors


def validate_evidence_manifest(path: str, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate a decoded evidence manifest without network access."""
    errors: list[dict[str, Any]] = []
    if not isinstance(manifest, Mapping):
        return [_error(path, "invalid_manifest", "manifest root must be an object")]

    unknown = sorted(set(manifest) - {"manifest_version", "chapter_id", "entries"})
    if unknown:
        errors.append(
            _error(
                path,
                "unknown_manifest_field",
                "manifest contains unknown top-level fields: " + ", ".join(unknown),
                fields=unknown,
            )
        )
    version = manifest.get("manifest_version")
    if not _is_int(version) or version != 1:
        errors.append(_error(path, "invalid_manifest_version", "manifest_version must be integer 1"))
    chapter_id = manifest.get("chapter_id")
    if not isinstance(chapter_id, str) or not _ID_RE.fullmatch(chapter_id):
        errors.append(_error(path, "invalid_chapter_id", "chapter_id must be a stable chapter id"))
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append(_error(path, "invalid_entries", "entries must be a non-empty array"))
        return errors

    seen_experiments: set[str] = set()
    for index, entry in enumerate(entries):
        prefix = f"entries[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(_error(path, "invalid_evidence_entry", f"{prefix} must be an object", entry=index))
            continue
        unknown = sorted(set(entry) - set(EVIDENCE_FIELDS))
        if unknown:
            errors.append(_error(path, "unknown_evidence_field", f"{prefix} contains unknown fields: {', '.join(unknown)}", entry=index, fields=unknown))
        if not _is_nonempty_string(entry.get("claim")):
            errors.append(_error(path, "invalid_claim", f"{prefix}.claim must be a non-empty string", entry=index))
        if entry.get("type") not in EVIDENCE_TYPES:
            errors.append(_error(path, "invalid_evidence_type", f"{prefix}.type is not a supported evidence classification", entry=index))
        if not isinstance(entry.get("source_url"), str) or not _URL_RE.fullmatch(entry["source_url"]):
            errors.append(_error(path, "invalid_source_url", f"{prefix}.source_url must be an http(s) URL", entry=index))
        if not _is_nonempty_string(entry.get("version")) or entry.get("version", "").casefold() in _BRANCH_NAMES:
            errors.append(_error(path, "invalid_version", f"{prefix}.version must identify a pinned release/commit", entry=index))
        experiment_id = entry.get("experiment_id")
        if experiment_id is not None and (not isinstance(experiment_id, str) or not _is_nonempty_string(experiment_id)):
            errors.append(_error(path, "invalid_experiment_id", f"{prefix}.experiment_id must be a string or null", entry=index))
        elif isinstance(experiment_id, str):
            if experiment_id in seen_experiments:
                errors.append(_error(path, "duplicate_experiment_id", f"{prefix}.experiment_id is repeated", entry=index, experiment_id=experiment_id))
            seen_experiments.add(experiment_id)
        if not _is_nonempty_string(entry.get("limitation")):
            errors.append(_error(path, "invalid_limitation", f"{prefix}.limitation must state what the evidence cannot establish", entry=index))
        if not _is_iso_date(entry.get("review_date")):
            errors.append(_error(path, "invalid_review_date", f"{prefix}.review_date must be an ISO date (YYYY-MM-DD)", entry=index))
    return errors


def validate_evidence_file(path: str, text: str) -> list[dict[str, Any]]:
    """Decode JSON text and validate it, returning a parse error instead of raising."""
    import json

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return [_error(path, "invalid_json", f"manifest is not valid JSON: {exc.msg}", line=exc.lineno, column=exc.colno)]
    return validate_evidence_manifest(path, value)
