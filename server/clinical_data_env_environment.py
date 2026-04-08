"""
Clinical Data Standardization Environment.

Trains AI agents to perform pharmaceutical clinical trial data standardization:
  Task 1 (Easy)   — Raw EDC → CDISC SDTM DM mapping
  Task 2 (Medium) — SDTM AE validation + violation correction
  Task 3 (Hard)   — SDTM LB → ADAM ADLB derivation (HbA1c endpoint)
"""

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Resolve data and graders relative to this file
_ENV_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ENV_ROOT))

from graders import grade_task1, grade_task2, grade_task3  # noqa: E402

try:
    from ..models import ClinicalAction, ClinicalObservation
except ImportError:
    from models import ClinicalAction, ClinicalObservation


TASK_NAMES = {
    1: "task1_edc_to_sdtm",
    2: "task2_sdtm_validation",
    3: "task3_sdtm_to_adam",
}

TASK_ORDER = [1, 2, 3]


def _get_benchmark_set() -> str:
    """Resolve the active benchmark set at runtime."""
    return os.getenv("BENCHMARK_SET", "toy").strip().lower()


def _load_all_data() -> dict:
    data: dict = {}
    data_dir = _ENV_ROOT / "data"
    benchmark_set = _get_benchmark_set()
    if benchmark_set != "toy":
        candidate_dir = data_dir / benchmark_set
        if not candidate_dir.exists():
            raise FileNotFoundError(
                f"Benchmark set '{benchmark_set}' was requested but {candidate_dir} does not exist."
            )
        data_dir = candidate_dir

    for task_id in TASK_ORDER:
        path = data_dir / f"task{task_id}_cases.json"
        with open(path, encoding="utf-8") as f:
            data[task_id] = json.load(f)
    return data


class ClinicalDataEnvironment(Environment):
    """
    OpenEnv environment for clinical trial data standardization.

    Each episode covers one task (5 cases). Calling reset() cycles through
    tasks 1 → 2 → 3 → 1 → ... so an inference script can run three episodes
    to evaluate all tasks.

    Reward: per-case score (0.0–1.0) during the episode; final step returns
    the task average reward with done=True.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_ATTEMPTS_PER_CASE: int = 2  # first attempt + 1 retry

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._data = _load_all_data()
        self._task_cycle_idx: int = -1   # incremented on first reset → starts at 0 → Task 1
        self._current_task: int = 0
        self._cases: list = []
        self._current_case_idx: int = 0
        self._episode_rewards: list[float] = []
        self._current_case_attempts: int = 0  # attempts on the current case

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> ClinicalObservation:
        """Advance to the next task and return the first case observation."""
        self._task_cycle_idx = (self._task_cycle_idx + 1) % len(TASK_ORDER)
        self._current_task = TASK_ORDER[self._task_cycle_idx]
        self._cases = self._data[self._current_task]["cases"]
        self._current_case_idx = 0
        self._episode_rewards = []
        self._current_case_attempts = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return self._make_observation(
            case=self._cases[0],
            feedback="",
            reward=0.0,
            done=False,
        )

    def step(self, action: ClinicalAction) -> ClinicalObservation:  # type: ignore[override]
        """Grade the submitted action and return the next case (or done)."""
        self._state.step_count += 1

        if self._current_case_idx >= len(self._cases):
            # Already done — idempotent terminal observation
            avg = sum(self._episode_rewards) / len(self._episode_rewards) if self._episode_rewards else 0.0
            return ClinicalObservation(
                task_id=self._current_task,
                task_name=TASK_NAMES[self._current_task],
                task_description="Episode already complete. Call reset() to start a new task.",
                input_data=None,
                study_context="",
                feedback="No action graded — episode was already done.",
                case_number=len(self._cases),
                total_cases=len(self._cases),
                task_score=round(avg, 4),
                done=True,
                reward=round(avg, 4),
            )

        case = self._cases[self._current_case_idx]
        score, feedback = self._grade(action, case)
        self._current_case_attempts += 1

        # Allow one retry if score is not perfect and within attempt limit
        if score < 1.0 and self._current_case_attempts < self.MAX_ATTEMPTS_PER_CASE:
            obs = self._make_observation(
                case=case,
                feedback=feedback,
                reward=round(score, 4),
                done=False,
            )
            obs.case_attempt = self._current_case_attempts + 1
            return obs

        # Case complete — apply 15% penalty if a retry was used
        attempt_penalty = 1.0 if self._current_case_attempts == 1 else 0.85
        final_score = round(score * attempt_penalty, 4)
        self._episode_rewards.append(final_score)
        self._current_case_idx += 1
        self._current_case_attempts = 0

        done = self._current_case_idx >= len(self._cases)
        avg_reward = sum(self._episode_rewards) / len(self._episode_rewards)

        if done:
            return ClinicalObservation(
                task_id=self._current_task,
                task_name=TASK_NAMES[self._current_task],
                task_description="Task complete. All cases evaluated.",
                input_data=None,
                study_context="",
                feedback=feedback,
                case_number=self._current_case_idx,
                total_cases=len(self._cases),
                task_score=round(avg_reward, 4),
                done=True,
                reward=round(avg_reward, 4),
            )

        next_case = self._cases[self._current_case_idx]
        return self._make_observation(
            case=next_case,
            feedback=feedback,
            reward=round(final_score, 4),
            done=False,
        )

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_observation(
        self,
        case: dict,
        feedback: str,
        reward: float,
        done: bool,
    ) -> ClinicalObservation:
        return ClinicalObservation(
            task_id=self._current_task,
            task_name=TASK_NAMES[self._current_task],
            task_description=case["task_description"],
            input_data=case["input_data"],
            study_context=case["study_context"],
            feedback=feedback,
            case_number=self._current_case_idx + 1,
            total_cases=len(self._cases),
            task_score=round(
                sum(self._episode_rewards) / len(self._episode_rewards)
                if self._episode_rewards else 0.0,
                4,
            ),
            case_attempt=self._current_case_attempts + 1,
            done=done,
            reward=reward,
        )

    def _grade(self, action: ClinicalAction, case: dict) -> tuple[float, str]:
        gt = case["ground_truth"]
        output = action.output_data
        task_id = self._current_task

        if task_id == 1:
            return grade_task1(output, gt)
        elif task_id == 2:
            return grade_task2(output, gt)
        elif task_id == 3:
            return grade_task3(output, gt)
        return 0.0, f"Unknown task id: {task_id}"
