"""Fixture-driven contract tests for the phase-two teaching template.

The tests intentionally exercise the public validator as a reader-facing contract:
metadata validation is deterministic and does not pretend that heading counts prove
teaching quality.  They can be run from the repository root with ``python -m
unittest scripts/test_phase2_template.py``.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from phase2_template import validate_chapter_file, validate_evidence_file, validate_evidence_manifest
except ModuleNotFoundError:  # ``python -m unittest scripts/test_phase2_template.py``
    from scripts.phase2_template import validate_chapter_file, validate_evidence_file, validate_evidence_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "scripts" / "fixtures" / "phase2_template"


def fixture_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def fixture_json(name: str) -> dict:
    return json.loads(fixture_text(name))


VALID_FRONTMATTER = """---
id: phase2-01-evidence-language
title: 证据语言与 AI Infrastructure 对象
description: 从一个具体故障出发，学习如何把观察变成可复核的证据。
slug: /phase2/chapters/01-evidence-language
sidebar_position: 101
phase: 2
chapter_number: 1
level: foundation
prerequisites: []
learning_objectives:
  - 能用直觉例子区分事实、测量和推断
  - 能给一个公式写出变量、单位和失效条件
  - 能运行复现实验并记录原始输出
paper_count: 1
source_commits:
  - openai/example@0123456789abcdef0123456789abcdef01234567
lab_paths:
  - labs/phase2/evidence_language.py
last_verified: 2026-10-06
---
"""

BEGINNER_FIRST_BODY = """# 证据语言与 AI Infrastructure 对象

## 问题边界
两个请求为什么会得到不同的尾延迟？本节先限定我们要解释的现象。

## 直觉模型
先画一个只有两个请求的时间线，再给出最小的可手算例子。

## 最小例子
请求 A 等待 2 ms，请求 B 等待 8 ms；我们先算 p50 和 p95。

## 正式定义与推导
定义变量、单位和公式，然后说明推导依赖的假设。

## 机制与源码入口
沿着 request → scheduler → worker 的调用链检查状态变化。

## 失效边界
如果把独立同分布或固定硬件的假设拿掉，结论会怎样？

## 可运行实验
基线、变量、控制变量、命令、原始输出、统计方法和误差来源均有记录。

## 失败诊所
一个看似健康的指标如何掩盖排队问题？

## 理解检查与答案
1. 哪个观察支持这个结论？答案：时间线中的等待区间。
2. 哪个假设最脆弱？答案：固定服务时间。

## 练习与研究问题
改变请求到达率，记录 p50/p95/p99，并写出不能推断的部分。

## 来源地图与复现清单
见证据清单；按固定版本运行实验命令。
"""

VALID_CHAPTER = VALID_FRONTMATTER + BEGINNER_FIRST_BODY

VALID_MANIFEST = {
    "manifest_version": 1,
    "chapter_id": "phase2-01-evidence-language",
    "entries": [
        {
            "claim": "排队会使高分位延迟显著高于中位数",
            "type": "experiment_measurement",
            "source_url": "https://example.com/lab/evidence-language",
            "version": "0123456789abcdef0123456789abcdef01234567",
            "experiment_id": "exp-evidence-001",
            "limitation": "CPU 模型只说明数量级，不能证明 GPU serving 的绝对延迟",
            "review_date": "2026-10-06",
        },
        {
            "claim": "p95 是按排序后第 95 个百分位定义的统计量",
            "type": "definition",
            "source_url": "https://www.rfc-editor.org/rfc/rfc2330",
            "version": "RFC 2330",
            "experiment_id": None,
            "limitation": "百分位插值约定可能因工具而异",
            "review_date": "2026-10-06",
        },
    ],
}


class Phase2TemplateContractTests(unittest.TestCase):
    def test_valid_beginner_first_fixture_passes(self) -> None:
        result = validate_chapter_file("fixtures/valid.md", fixture_text("valid.md"))
        self.assertEqual(result, [])

    def test_beginner_first_fixture_is_checked_as_a_real_file(self) -> None:
        result = validate_chapter_file("fixtures/beginner-first.md", fixture_text("beginner-first.md"))
        self.assertEqual(result, [])

    def test_metadata_only_fixture_reports_editorial_gap_separately(self) -> None:
        # Metadata can be valid before an author has written the teaching body.
        result = validate_chapter_file("fixtures/metadata-only.md", VALID_FRONTMATTER + "# 标题\n")
        classes = {error["class"] for error in result}
        self.assertIn("beginner_sequence", classes)
        self.assertNotIn("missing_frontmatter", classes)

    def test_invalid_fixture_has_stable_error_order_and_classes(self) -> None:
        invalid = """---
id: phase2-01-invalid
phase: one
chapter_number: 0
level: unknown
prerequisites: nope
learning_objectives: []
paper_count: -1
source_commits: ../not-a-commit
lab_paths:
  - /absolute/path
last_verified: yesterday
---
# 问题边界
"""
        result = validate_chapter_file("fixtures/invalid.md", fixture_text("invalid.md"))
        self.assertEqual(
            [error["class"] for error in result],
            [
                "missing_legacy_frontmatter",
                "invalid_phase",
                "invalid_chapter_number",
                "invalid_level",
                "invalid_prerequisites",
                "invalid_learning_objectives",
                "invalid_paper_count",
                "invalid_source_commits",
                "invalid_lab_paths",
                "invalid_last_verified",
                "beginner_sequence",
            ],
        )
        self.assertEqual([error["path"] for error in result], ["fixtures/invalid.md"] * len(result))

    def test_manifest_fixture_passes(self) -> None:
        self.assertEqual(validate_evidence_manifest("fixtures/valid-evidence.json", fixture_json("valid-evidence.json")), [])

    def test_manifest_errors_are_deterministic_and_explain_missing_provenance(self) -> None:
        invalid = {
            "manifest_version": 1,
            "chapter_id": "phase2-01-evidence-language",
            "entries": [
                {
                    "claim": "",
                    "type": "guess",
                    "source_url": "ftp://invalid",
                    "version": "",
                    "experiment_id": 3,
                    "limitation": "",
                    "review_date": "2026-99-99",
                }
            ],
        }
        result = validate_evidence_manifest("fixtures/invalid-evidence.json", invalid)
        self.assertEqual(
            [error["class"] for error in result],
            [
                "invalid_claim",
                "invalid_evidence_type",
                "invalid_source_url",
                "invalid_version",
                "invalid_experiment_id",
                "invalid_limitation",
                "invalid_review_date",
            ],
        )

    def test_manifest_parser_reports_invalid_json_without_network_access(self) -> None:
        result = validate_evidence_file("fixtures/broken-evidence.json", "{broken")
        self.assertEqual([error["class"] for error in result], ["invalid_json"])

    def test_manifest_rejects_duplicate_experiment_entries(self) -> None:
        duplicate = fixture_json("valid-evidence.json")
        duplicate["entries"].append(dict(duplicate["entries"][0]))
        result = validate_evidence_manifest("fixtures/duplicate-evidence.json", duplicate)
        self.assertEqual([error["class"] for error in result], ["duplicate_experiment_id"])

    def test_schema_declares_exact_evidence_fields(self) -> None:
        schema = json.loads((REPO_ROOT / "docs/phase2/evidence-manifest.schema.json").read_text(encoding="utf-8"))
        evidence = schema["$defs"]["evidence"]
        self.assertEqual(
            set(evidence["required"]),
            {"claim", "type", "source_url", "version", "experiment_id", "limitation", "review_date"},
        )
        self.assertEqual(evidence["additionalProperties"], False)


if __name__ == "__main__":
    unittest.main()
