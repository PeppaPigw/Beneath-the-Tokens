---
id: phase2-02-beginner-first
title: 新人优先的机制导读
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
# 新人优先的机制导读

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
