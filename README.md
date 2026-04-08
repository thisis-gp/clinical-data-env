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

This environment covers three common clinical programming workflows:

1. Convert raw EDC demographic data to SDTM DM.
2. Validate SDTM AE records against CDISC terminology rules.
3. Derive ADaM ADLB records from SDTM laboratory data.

Domain focus: Type 2 Diabetes, with HbA1c as the primary efficacy endpoint.

## Tasks

| Task | Name | Difficulty | Description |
|------|------|------------|-------------|
| 1 | `task1_edc_to_sdtm` | Easy | Convert raw EDC demographic records into SDTM DM fields |
| 2 | `task2_sdtm_validation` | Medium | Detect AE terminology violations and propose corrected values |
| 3 | `task3_sdtm_to_adam` | Hard | Derive ADLB values such as `BASE`, `CHG`, and `PCHG` |

Each task contains 5 cases. One episode covers one full task.

## Action Space

`ClinicalAction`

| Field | Type | Description |
|------|------|-------------|
| `task_id` | `int` | Task identifier: 1, 2, or 3 |
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
| `study_context` | `str` | CDISC rules or SAP derivation rules |
| `feedback` | `str` | Grader feedback from the prior step |
| `case_number` | `int` | Current case number |
| `total_cases` | `int` | Number of cases in the task |
| `task_score` | `float` | Running average score |
| `reward` | `float` | Reward for the most recent step |
| `done` | `bool` | Whether the task is complete |

## Reward Functions

### Task 1

Field match score with a small penalty for hallucinated fields.

### Task 2

`0.5 * F1(violation detection) + 0.5 * correction accuracy`

### Task 3

Exact-match scoring for string fields and tolerance-based scoring for numeric fields.

## Setup

### Local Development

```bash
cd clinical_data_env
uv sync
uv run server
```

In another terminal:

```bash
HF_TOKEN=your_token python inference.py
```

### Docker

```bash
docker build -t clinical_data_env -f server/Dockerfile .
docker run -p 8000:8000 clinical_data_env
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | - | API key for LLM calls |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `ENV_BASE_URL` | No | `http://localhost:8000` | OpenEnv server URL |
| `BENCHMARK_SET` | No | `toy` | Case set to load: `toy` or `pharmaverse` |

## Benchmark Sets

- `toy`: the original hand-authored demo cases under `data/`
- `pharmaverse`: a second, more realistic benchmark set generated from the cloned `pharmaversesdtm` data under `data/pharmaverse/`

Example:

```powershell
$env:BENCHMARK_SET="pharmaverse"
.\.venv\Scripts\python.exe -m server.app --port 8000
```

## Local `.env` Setup

The project can load settings from a local [`.env`](C:/Users/am400/Desktop/Projects/project005/clinical_data_env/.env) file automatically.

1. Fill in [`.env`](C:/Users/am400/Desktop/Projects/project005/clinical_data_env/.env) with your local values.
2. Start the server or run inference without exporting everything manually.

Example `.env` fields:

```text
HF_TOKEN=your_token_here
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
|   `-- task3_cases.json
|-- graders/
|   |-- grader_task1.py
|   |-- grader_task2.py
|   `-- grader_task3.py
`-- server/
    |-- app.py
    |-- clinical_data_env_environment.py
    |-- Dockerfile
    `-- requirements.txt
```
