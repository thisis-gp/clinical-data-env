# Benchmark Summary

## Overview

This project now supports two benchmark sets:

| Benchmark Set | Purpose | Status |
|---|---|---|
| `toy` | Controlled smoke-test and demo validation set | Stable |
| `pharmaverse` | More realistic data-derived evaluation set | Active |

## Latest Reference Results

| Benchmark Set | Model | Task 1 | Task 2 | Task 3 | Overall |
|---|---|---:|---:|---:|---:|
| `toy` | `Qwen/Qwen2.5-72B-Instruct` | 0.9600 | 1.0000 | 1.0000 | 0.9867 |
| `pharmaverse` | `Qwen/Qwen2.5-72B-Instruct` | 1.0000 | 0.9709 | 0.8683 | 0.9464 |

## Interpretation

- The `toy` benchmark is useful for validating that the environment, client, grading loop, and prompt formatting all work.
- The `pharmaverse` benchmark is more informative and exposes realistic model weaknesses.
- On the current `pharmaverse` run, the biggest losses come from:
  - Task 1: subject identifier construction and day/month date parsing
  - Task 2: exact sponsor terminology matching
  - Task 3: exact ADaM-style field and flag conventions

## Recommended Demo Message

Use `toy` to show that the system works end to end.

Use `pharmaverse` to show that the benchmark is not trivial and that the model still makes clinically meaningful transformation errors that can be measured quantitatively.

## Reproducing

Server:

```powershell
.\.venv\Scripts\python.exe -m server.app
```

Inference:

```powershell
python inference.py
```

Set the active benchmark via `.env`:

```text
BENCHMARK_SET=toy
```

or

```text
BENCHMARK_SET=pharmaverse
```
