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

Trains AI agents to standardize pharmaceutical clinical trial data from raw EDC records to CDISC SDTM and ADaM outputs.

## Overview

This environment covers four common clinical programming workflows:

1. Convert raw EDC demographic data to SDTM DM.
2. Validate SDTM AE records against CDISC terminology rules.
3. Derive ADaM ADLB records from SDTM laboratory data.
4. Detect cross-domain inconsistencies across SDTM DM, EX, CM, AE, and DS records.

Domain focus: Type 2 Diabetes, with HbA1c as the primary efficacy endpoint.

## Tasks

| Task | Name | Difficulty | Description |
|------|------|------------|-------------|
| 1 | `task1_edc_to_sdtm` | Easy | Convert raw EDC demographic records into SDTM DM fields |
| 2 | `task2_sdtm_validation` | Medium | Detect AE terminology violations and propose corrected values |
| 3 | `task3_sdtm_to_adam` | Hard | Derive ADLB values such as `BASE`, `CHG`, and `PCHG` |
| 4 | `task4_cross_domain_validation` | Hard | Identify cross-domain inconsistencies such as first-dose mismatches, prohibited medication timing, and orphan SAEs |

The `toy` benchmark contains 5 cases per task. The `pharmaverse` benchmark contains 10 cases each for Tasks 1-3 and 5 cases for Task 4. One episode covers one full task, with up to two attempts per case.

## Action Space

`ClinicalAction`

| Field | Type | Description |
|------|------|-------------|
| `task_id` | `int` | Task identifier: 1, 2, 3, or 4 |
| `output_data` | `dict | list` | Task answer payload |
| `reasoning` | `str` | Optional explanation for logging |

## Observation Space

`ClinicalObservation`

| Field | Type | Description |
|------|------|-------------|
| `task_id` | `int` | Current task identifier |
| `task_name` | `str` | Task name |
| `task_description` | `str` | Instructions for the current case |
| `input_data` | `dict | list` | Case payload |
| `study_context` | `str` | CDISC rules or protocol instructions |
| `feedback` | `str` | Grader feedback from the prior step |
| `case_number` | `int` | Current case number |
| `total_cases` | `int` | Number of cases in the task |
| `task_score` | `float` | Running average score |
| `case_attempt` | `int` | Current attempt number for the case |
| `detection_score` | `float` | Detection or structure-finding sub-score from the previous step |
| `correction_score` | `float` | Value-accuracy sub-score from the previous step |
| `field_scores` | `dict[str, float]` | Per-field or per-issue breakdown from the previous step |
| `sub_scores` | `dict` | Full component breakdown from the previous step |
| `episode_summary` | `dict` | Per-case summary populated when `done=true` |
| `difficulty_breakdown` | `dict[str, float]` | Average score by difficulty when `done=true` |
| `reward` | `float` | Reward for the most recent step |
| `done` | `bool` | Whether the task is complete |

## Reward Functions

### Task 1

Field match score with a small penalty for hallucinated fields.

### Task 2

`0.5 * F1(violation detection) + 0.5 * correction accuracy`

### Task 3

Exact-match scoring for string fields and field-specific tolerances for numeric fields.

### Task 4

`0.6 * F1(issue detection) + 0.4 * issue detail accuracy`

## RL-Friendly Features

- Multi-turn episodes with one retry per case
- Retry penalty on second-attempt success
- Structured sub-scores for detection, correction, and per-field breakdowns
- Terminal `episode_summary` and `difficulty_breakdown` outputs for training analysis

## Setup

### Local Development

```bash
cd clinical_data_env
uv sync
uv run server
```

In another terminal:

```bash
python inference.py
```

### Docker

```bash
docker build -t clinical_data_env -f server/Dockerfile .
docker run -p 8000:8000 clinical_data_env
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | No | - | Optional provider API key alias |
| `API_KEY` | No | - | API key used by the inference script when `HF_TOKEN` is empty |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `ENV_BASE_URL` | No | `http://localhost:8000` | OpenEnv server URL |
| `BENCHMARK_SET` | No | `toy` | Case set to load: `toy` or `pharmaverse` |
| `SERVER_PORT` | No | `8000` | Local API port for the environment server |

## Benchmark Sets

- `toy`: the original hand-authored demo cases under `data/`
- `pharmaverse`: a more realistic benchmark under `data/pharmaverse/` with harder Task 1 ambiguity, zero-violation Task 2 cases, less explicit Task 3 derivations, and cross-domain Task 4 cases

Example:

```powershell
$env:BENCHMARK_SET="pharmaverse"
.\.venv\Scripts\python.exe -m server.app
```

## Local `.env` Setup

The project can load settings from a local [`.env`](C:/Users/am400/Desktop/Projects/project005/clinical_data_env/.env) file automatically.

1. Fill in [`.env`](C:/Users/am400/Desktop/Projects/project005/clinical_data_env/.env) with your local values.
2. Start the server or run inference without exporting everything manually.

Example `.env` fields:

```text
API_KEY=your_provider_key_here
BENCHMARK_SET=pharmaverse
ENV_BASE_URL=http://localhost:8001
SERVER_PORT=8001
```

## Project Structure

```text
clinical_data_env/
|-- models.py
|-- client.py
|-- inference.py
|-- openenv.yaml
|-- pyproject.toml
|-- data/
|   |-- task1_cases.json
|   |-- task2_cases.json
|   |-- task3_cases.json
|   `-- task4_cases.json
|-- graders/
|   |-- grader_task1.py
|   |-- grader_task2.py
|   |-- grader_task3.py
|   `-- grader_task4.py
|-- tests/
|   |-- conftest.py
|   `-- test_graders.py
`-- server/
    |-- app.py
    |-- clinical_data_env_environment.py
    |-- Dockerfile
    `-- requirements.txt
```
