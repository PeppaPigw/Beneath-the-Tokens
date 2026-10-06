---
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
# Evidence language

## Why this problem exists
Two requests can share a mean latency while their tail waits differ.

## Mental model
Draw the queue before naming a scheduler.

## Minimal example
Request A waits 2 ms; request B waits 8 ms.

## Formal definition
Define the percentile, variables, and units before using the formula.

## System mechanism
Trace request → scheduler → worker at the pinned commit.

## Failure boundary
Drop the fixed-service-time assumption and inspect the tail again.

## Measured experiment
Run the CPU fallback with a baseline, repeats, and raw output.
