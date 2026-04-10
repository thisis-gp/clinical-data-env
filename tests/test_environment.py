"""Tests for ClinicalDataEnvironment observation fields."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.clinical_data_env_environment import ClinicalDataEnvironment
from models import ClinicalAction


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_env() -> ClinicalDataEnvironment:
    """Return a freshly instantiated environment."""
    return ClinicalDataEnvironment()


def perfect_action_for_task(task_id: int) -> ClinicalAction:
    """Return a plausible (not necessarily perfect) action for the given task."""
    if task_id == 1:
        output = {
            "USUBJID": "CDISCPILOT01-01-001",
            "AGE": 52,
            "SEX": "M",
            "RACE": "WHITE",
            "RFSTDTC": "2024-06-15",
            "COUNTRY": "USA",
        }
    elif task_id == 2:
        output = {"violations": []}
    elif task_id == 3:
        output = [
            {
                "USUBJID": "CDISCPILOT01-01-001",
                "VISIT": "SCREENING",
                "PARAM": "Hemoglobin A1c (HbA1c) [%]",
                "AVAL": 8.4,
                "BASE": 8.4,
                "CHG": 0.0,
                "PCHG": 0.0,
                "ABLFL": "Y",
                "ANL01FL": "",
            }
        ]
    else:  # task_id == 4
        output = {"issues": []}
    return ClinicalAction(task_id=task_id, output_data=output, reasoning="test")


def cycle_env_to_task(env: ClinicalDataEnvironment, target_task: int) -> None:
    """Reset the env until it lands on target_task."""
    for _ in range(4):  # at most 4 resets to cycle through all tasks
        obs = env.reset()
        if obs.task_id == target_task:
            return
    raise RuntimeError(f"Could not reach task {target_task} within 4 resets")


# ── pre_step_hints ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task_id", [1, 2, 3, 4])
def test_pre_step_hints_is_nonempty_list(task_id):
    """pre_step_hints must be a non-empty list for every task on the first case."""
    env = make_env()
    cycle_env_to_task(env, task_id)
    obs = env.reset() if env._current_task != task_id else env._make_observation(
        env._cases[0], feedback="", reward=0.0, done=False
    )
    # Ensure we are looking at the correct task
    cycle_env_to_task(env, task_id)
    obs = env._make_observation(env._cases[0], feedback="", reward=0.0, done=False)
    assert isinstance(obs.pre_step_hints, list), "pre_step_hints should be a list"
    assert len(obs.pre_step_hints) > 0, f"pre_step_hints should not be empty for task {task_id}"


def test_task2_hints_do_not_leak_ground_truth_cleanliness():
    """Task 2 hints must be derived from input only, not from the hidden answer key."""
    env = make_env()
    cycle_env_to_task(env, 2)

    clean_case = next(case for case in env._cases if not case["ground_truth"]["violations"])
    violating_case = next(case for case in env._cases if case["ground_truth"]["violations"])

    clean_hints = env._generate_pre_step_hints(clean_case)
    violating_hints = env._generate_pre_step_hints(violating_case)

    forbidden_phrase = "already be SDTM-clean"
    assert all(forbidden_phrase not in hint for hint in clean_hints)
    assert all(forbidden_phrase not in hint for hint in violating_hints)


# ── action_history ────────────────────────────────────────────────────────────

def test_action_history_starts_empty():
    """action_history must be empty on reset (before any step)."""
    env = make_env()
    obs = env.reset()
    assert obs.action_history == [], "action_history should be empty after reset"


def test_action_history_grows_after_step():
    """action_history must contain one entry after one step on a case."""
    env = make_env()
    obs = env.reset()
    task_id = obs.task_id
    action = perfect_action_for_task(task_id)
    obs2 = env.step(action)
    # If the step triggered a retry (score < 1 and attempts < MAX), history grows
    # If the step advanced to next case, history was reset — check either way
    # The observation returned when still on the same case should have history
    if obs2.case_number == obs.case_number and not obs2.done:
        assert len(obs2.action_history) >= 1, "action_history should grow after a step on the same case"


def test_action_history_resets_on_new_case():
    """action_history must be empty when the env advances to a new case."""
    env = make_env()
    obs = env.reset()
    task_id = obs.task_id

    # Submit a perfect (or good enough) answer to advance past case 1
    action = perfect_action_for_task(task_id)
    obs2 = env.step(action)

    if not obs2.done and obs2.case_number > obs.case_number:
        # We moved to a new case — history should be reset
        assert obs2.action_history == [], "action_history should be empty at the start of a new case"


# ── field_scores ──────────────────────────────────────────────────────────────

def test_field_scores_populated_after_step_task1():
    """field_scores should be a non-empty dict in the observation after a Task 1 step."""
    env = make_env()
    cycle_env_to_task(env, 1)
    action = ClinicalAction(
        task_id=1,
        output_data={
            "USUBJID": "CDISCPILOT01-01-001",
            "AGE": 52,
            "SEX": "M",
            "RACE": "WHITE",
            "RFSTDTC": "2024-06-15",
            "COUNTRY": "USA",
        },
        reasoning="test",
    )
    obs = env.step(action)
    assert isinstance(obs.field_scores, dict), "field_scores should be a dict"
    assert len(obs.field_scores) > 0, "field_scores should be populated after a Task 1 step"


def test_field_scores_populated_after_step_task2():
    """field_scores should be present in the observation after a Task 2 step."""
    env = make_env()
    cycle_env_to_task(env, 2)
    # Submit a clearly wrong answer so grader runs and populates sub_scores
    action = ClinicalAction(
        task_id=2,
        output_data={"violations": [{"field": "AESEV", "issue": "wrong case", "corrected_value": "MILD"}]},
        reasoning="test",
    )
    obs = env.step(action)
    assert isinstance(obs.field_scores, dict), "field_scores should be a dict after Task 2 step"


# ── episode_summary and difficulty_breakdown when done=True ──────────────────

def _run_task_to_completion(task_id: int) -> "ClinicalDataEnvironment":
    """Run through all cases of a task and return the final done observation."""
    env = make_env()
    cycle_env_to_task(env, task_id)
    action = perfect_action_for_task(task_id)
    obs = None
    # Submit the same action up to (cases * MAX_ATTEMPTS + buffer) times
    max_iters = 30
    for _ in range(max_iters):
        obs = env.step(action)
        if obs.done:
            break
    return obs


def test_episode_summary_populated_when_done():
    """episode_summary must be a non-empty dict when done=True."""
    obs = _run_task_to_completion(1)
    assert obs.done, "Expected done=True after running all cases"
    assert isinstance(obs.episode_summary, dict), "episode_summary should be a dict"
    assert len(obs.episode_summary) > 0, "episode_summary should be populated when done=True"


def test_difficulty_breakdown_populated_when_done():
    """difficulty_breakdown must be a non-empty dict when done=True."""
    obs = _run_task_to_completion(1)
    assert obs.done, "Expected done=True after running all cases"
    assert isinstance(obs.difficulty_breakdown, dict), "difficulty_breakdown should be a dict"
    assert len(obs.difficulty_breakdown) > 0, "difficulty_breakdown should be populated when done=True"


def test_episode_summary_keys_are_case_ids():
    """episode_summary keys should match case IDs from the task."""
    env = make_env()
    cycle_env_to_task(env, 1)
    expected_case_ids = {case["case_id"] for case in env._cases}
    obs = _run_task_to_completion(1)
    assert obs.done
    for key in obs.episode_summary:
        assert isinstance(key, str), "episode_summary keys should be strings (case IDs)"
        assert key in expected_case_ids, f"Unexpected key in episode_summary: {key}"


def test_difficulty_breakdown_values_are_floats():
    """difficulty_breakdown values must be floats in (0, 1)."""
    obs = _run_task_to_completion(1)
    assert obs.done
    for difficulty, avg_score in obs.difficulty_breakdown.items():
        assert isinstance(avg_score, float), f"difficulty_breakdown[{difficulty}] should be float"
        assert 0 < avg_score < 1, f"difficulty_breakdown[{difficulty}] = {avg_score} not in (0, 1)"
