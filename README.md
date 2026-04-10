---
title: Clinical Data Standardization Environment
emoji: "🧬"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Clinical Data Standardization Environment

An OpenEnv benchmark and training environment for clinical trial data standardization, spanning raw EDC mapping, SDTM validation, ADaM derivation, and cross-domain consistency checks.

## Why This Is Different

Most benchmark environments stop at extraction or simple validation. This environment covers the actual downstream workflow clinical data teams use before analysis and submission:

1. Raw EDC to SDTM DM mapping
2. SDTM AE validation and correction
3. SDTM LB to ADaM ADLB derivation
4. Cross-domain consistency checks across DM, EX, CM, AE, and DS

The ADaM derivation task is the most distinctive piece. It forces the agent to identify baseline rows, derive `BASE`, `CHG`, and `PCHG`, preserve analysis flags, and handle clinically meaningful numeric tolerances instead of relying on naive exact matching.

## Benchmark Layout

### Tasks

| Task | Name | Difficulty | Description |
|------|------|------------|-------------|
| 1 | `task1_edc_to_sdtm` | Easy | Convert raw EDC demographic records into SDTM DM fields |
| 2 | `task2_sdtm_validation` | Medium | Detect AE terminology violations and propose corrected values |
| 3 | `task3_sdtm_to_adam` | Hard | Derive ADLB fields such as `BASE`, `CHG`, and `PCHG` |
| 4 | `task4_cross_domain_validation` | Very hard | Find cross-domain inconsistencies such as first-dose mismatches, prohibited medication timing, and orphan SAEs |

### Case Counts

| Benchmark | Task 1 | Task 2 | Task 3 | Task 4 |
|-----------|--------|--------|--------|--------|
| `toy` | 10 | 10 | 10 | 5 |
| `pharmaverse` | 10 | 10 | 10 | 10 |

Each episode covers one full task. Each case allows up to two attempts, with a retry penalty on second-attempt success.

## Observation Design

`ClinicalObservation` exposes both the task payload and RL-friendly training signals:

| Field | Purpose |
|-------|---------|
| `input_data` | Raw case payload |
| `study_context` | CDISC rules and protocol instructions |
| `pre_step_hints` | Deterministic, input-derived hints before the first action |
| `feedback` | Grader feedback from the previous attempt |
| `action_history` | Prior attempts already made on the current case |
| `detection_score` | Detection or structure-finding component |
| `correction_score` | Value-accuracy component |
| `field_scores` | Per-field or per-issue breakdown |
| `episode_summary` | Per-case summary once the task is complete |
| `difficulty_breakdown` | Average score by difficulty bucket when done |

This keeps the environment useful for both evaluation and training.

## Reward Design

### Task 1

Field match score with a small penalty for hallucinated fields.

### Task 2

`0.5 * F1(violation detection) + 0.5 * correction accuracy`

### Task 3

Exact-match scoring for string fields and field-specific tolerances for numeric derivations.

### Task 4

`0.6 * F1(issue detection) + 0.4 * issue detail accuracy`

All externally reported task scores are kept strictly inside `(0, 1)` for hackathon validator compliance.

## Baseline Performance

Full `pharmaverse` baseline run using `llama-3.3-70b-versatile` on April 10, 2026:

| Task | Score |
|------|-------|
| `task1_edc_to_sdtm` | `1.0000` |
| `task2_sdtm_validation` | `0.9122` |
| `task3_sdtm_to_adam` | `0.8566` |
| `task4_cross_domain_validation` | `0.4388` |
| Overall | `0.8019` |

Source log: [inference-pharmaverse-20260410-004317.log](/C:/Users/am400/Desktop/Projects/project005/clinical_data_env/logs/inference-pharmaverse-20260410-004317.log)

Interpretation:

- Task 1 is close to saturated and useful mainly for controlled ambiguity
- Task 2 remains strong but is no longer trivial because zero-violation cases exist
- Task 3 is a real reasoning task and a key differentiator
- Task 4 is currently the hardest task and the biggest remaining opportunity for improvement

## Benchmark Evidence

This repo now supports both full-benchmark and targeted-task evaluation so provider credits can be spent on the most informative task.

### Single-Port Local Benchmarking

Use one fixed local port for the environment. The launcher below reuses a healthy server on that port, otherwise starts one on the same port, runs inference, and then stops it.

```bash
python scripts/run_local_benchmark.py
```

To evaluate only the hardest task:

```bash
python scripts/run_local_benchmark.py --tasks task4
```

For single-task runs, the launcher now requires a fresh server on the configured port so task forcing is deterministic. If something is already listening on that port, stop it first and rerun.

`inference.py` also supports direct task targeting:

```bash
python inference.py --tasks task4
python inference.py --tasks 1,2,4
```

### Model Comparison Table

This is the section to keep updating as new open or free-tier model runs are completed.
The goal is not to list every provider we tried, but to show meaningful variance across a small, credible set of strong open models.

| Date | Provider | Model | Tasks | Task 1 | Task 2 | Task 3 | Task 4 | Overall | Status |
|------|----------|-------|-------|--------|--------|--------|--------|---------|--------|
| 2026-04-10 | Groq | `llama-3.3-70b-versatile` | all | `1.0000` | `0.9122` | `0.8566` | `0.4388` | `0.8019` | Valid full run |
| 2026-04-11 | Groq | `openai/gpt-oss-120b` | `task4` | - | - | - | `0.8357` | `0.8357` | Valid reasoning-focused Task 4 run |
| 2026-04-10 | Groq | `llama-3.1-8b-instant` | `task4` | - | - | - | `0.5972` | `0.5972` | Valid low-cost Task 4 run |
| Pending | Provider with valid access | `Qwen/Qwen2.5-72B-Instruct` | all or `task4` | - | - | - | TBD | - | Alternate open instruct baseline |

Recommended comparison set for judges:

- `llama-3.3-70b-versatile`: primary anchor baseline
- `openai/gpt-oss-120b`: reasoning-oriented comparison, with a valid `task4` score of `0.8357`
- `Qwen/Qwen2.5-72B-Instruct`: alternate open-model family
- one cheaper/free open instruct model: cost-efficiency comparison

Publishing guidance:

- Keep the main table limited to valid, reproducible runs
- Do not put provider-failure runs with `0.0100` fallback scores in the main comparison table
- If needed, document failed attempts separately as quota or permission issues rather than benchmark results

Attempted but not published in the main comparison table:

- `deepseek-r1-distill-llama-70b` on Groq: model is decommissioned by the provider, so it is no longer part of the active benchmark plan
- `Qwen/Qwen2.5-72B-Instruct` on Hugging Face Router: token lacked Inference Providers permission
- `openai/gpt-oss-20b` on Groq: promising partial `task4` run, but it hit a provider rate limit before completing all 10 cases

## 2026 Model Recommendations

Based on Groq's current official model catalog and deprecation guidance, the best Groq-side models to benchmark for this project in 2026 are:

- `llama-3.3-70b-versatile`: strong general open-weight baseline with a valid full pharmaverse run already recorded
- `openai/gpt-oss-120b`: Groq's current flagship open-weight reasoning model and the recommended replacement for several deprecated models
- `qwen/qwen3-32b`: current Qwen-family option on Groq for an alternate model family comparison
- `llama-3.1-8b-instant`: low-cost speed baseline that already produced a valid Task 4 run

These recommendations are an inference from Groq's current supported-models and deprecations pages rather than a claim that they are universally the best models overall.

## Local Dev Rate-Limit Guidance

Groq's current developer-plan model table lists model-specific TPM limits, so local benchmarking should be paced accordingly instead of assuming one universal rate limit.

Relevant current Groq limits from the official supported-models page:

- `llama-3.1-8b-instant`: `250K TPM`
- `llama-3.3-70b-versatile`: `300K TPM`
- `openai/gpt-oss-120b`: `250K TPM`
- `openai/gpt-oss-20b`: `250K TPM`
- `qwen/qwen3-32b`: `300K TPM`

The local inference script now includes conservative Groq request pacing plus 429 retry handling for local development. You can also override pacing manually with:

```bash
REQUEST_DELAY_SECONDS=2 python inference.py --tasks task4
```

or

```bash
RATE_LIMIT_MAX_RETRIES=3 python scripts/run_local_benchmark.py --tasks task4
```

## RL-Friendly Features

- Multi-turn episodes with one retry per case
- Retry-aware prompting through `case_attempt`
- Deterministic `pre_step_hints` derived only from observable input structure
- `action_history` so the agent can see what it already tried
- Structured sub-scores for richer training signal
- Terminal summaries with per-case and per-difficulty breakdowns

## Setup

### Local Development

```bash
cd clinical_data_env
uv sync
uv run server
```

Run inference in another terminal:

```bash
python inference.py
```

Run only selected tasks:

```bash
python inference.py --tasks task4
```

### Docker

```bash
docker build -t clinical_data_env -f server/Dockerfile .
docker run -p 8000:8000 clinical_data_env
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Optional provider API key alias |
| `API_KEY` | API key used by inference when `HF_TOKEN` is empty |
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `ENV_BASE_URL` | OpenEnv server URL |
| `BENCHMARK_SET` | Case set to load: `toy` or `pharmaverse` |
| `INFERENCE_TASKS` | Comma-separated task ids or names to run, or `all` |
| `SERVER_PORT` | Local API port for the environment server |

## Project Structure

```text
clinical_data_env/
|-- models.py
|-- client.py
|-- inference.py
|-- openenv.yaml
|-- pyproject.toml
|-- data/
|-- graders/
|-- tests/
`-- server/
```
