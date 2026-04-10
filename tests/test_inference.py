"""Tests for inference task selection helpers."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import inference


def test_parse_task_selection_defaults_to_all():
    assert inference.parse_task_selection("all") == inference.TASK_NAMES


def test_parse_task_selection_accepts_ids_and_names():
    selected = inference.parse_task_selection("4,task2,task4_cross_domain_validation")
    assert selected == [
        "task4_cross_domain_validation",
        "task2_sdtm_validation",
    ]


def test_parse_task_selection_rejects_unknown_task():
    with pytest.raises(ValueError):
        inference.parse_task_selection("task9")


def test_build_log_suffix_uses_selected_tasks():
    suffix = inference.build_log_suffix(["task4_cross_domain_validation"])
    assert suffix.endswith("t4-cross-domain-validation")


def test_parse_retry_after_seconds_from_message():
    exc = Exception("Rate limit reached. Please try again in 10.38s.")
    assert inference.parse_retry_after_seconds(exc) == 10.38


def test_groq_local_dev_delay_has_known_model_defaults():
    assert inference.groq_local_dev_delay("llama-3.1-8b-instant") > 0
