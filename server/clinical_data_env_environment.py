"""
Clinical Data Standardization Environment.

Tasks:
  1. Raw EDC -> CDISC SDTM DM mapping
  2. SDTM AE validation + correction
  3. SDTM LB -> ADAM ADLB derivation
  4. Cross-domain SDTM consistency validation
"""

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

_ENV_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ENV_ROOT))

from graders import grade_task1, grade_task2, grade_task3, grade_task4  # noqa: E402

try:
    from ..models import ClinicalAction, ClinicalObservation
except ImportError:
    from models import ClinicalAction, ClinicalObservation


TASK_NAMES = {
    1: "task1_edc_to_sdtm",
    2: "task2_sdtm_validation",
    3: "task3_sdtm_to_adam",
    4: "task4_cross_domain_validation",
}

TASK_ORDER = [1, 2, 3, 4]
SUBMISSION_SCORE_FLOOR = 0.01
SUBMISSION_SCORE_CAP = 0.99


def _get_benchmark_set() -> str:
    return os.getenv("BENCHMARK_SET", "toy").strip().lower()


def _load_all_data() -> dict:
    data: dict = {}
    data_dir = _ENV_ROOT / "data"
    benchmark_set = _get_benchmark_set()
    if benchmark_set != "toy":
        candidate_dir = data_dir / benchmark_set
        if not candidate_dir.exists():
            raise FileNotFoundError(f"Benchmark set '{benchmark_set}' was requested but {candidate_dir} does not exist.")
        data_dir = candidate_dir

    for task_id in TASK_ORDER:
        path = data_dir / f"task{task_id}_cases.json"
        with open(path, encoding="utf-8") as f:
            data[task_id] = json.load(f)
    return data


def _submission_safe_score(score: float) -> float:
    """Keep externally reported scores strictly inside (0, 1)."""
    return round(min(max(float(score), SUBMISSION_SCORE_FLOOR), SUBMISSION_SCORE_CAP), 4)


class ClinicalDataEnvironment(Environment):
    """OpenEnv environment for clinical trial data standardization."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_ATTEMPTS_PER_CASE: int = 2

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._data = _load_all_data()
        self._task_cycle_idx = -1
        self._current_task = 0
        self._cases: list[dict] = []
        self._current_case_idx = 0
        self._episode_rewards: list[float] = []
        self._current_case_attempts = 0
        self._last_sub_scores: dict = {}
        self._case_summaries: list[dict] = []

    def reset(self) -> ClinicalObservation:
        self._task_cycle_idx = (self._task_cycle_idx + 1) % len(TASK_ORDER)
        self._current_task = TASK_ORDER[self._task_cycle_idx]
        self._cases = self._data[self._current_task]["cases"]
        self._current_case_idx = 0
        self._episode_rewards = []
        self._current_case_attempts = 0
        self._last_sub_scores = {}
        self._case_summaries = []
        self._state = State(episode_id=str(uuid4()), step_count=0)
        return self._make_observation(self._cases[0], feedback="", reward=0.0, done=False)

    def step(self, action: ClinicalAction) -> ClinicalObservation:  # type: ignore[override]
        self._state.step_count += 1

        if self._current_case_idx >= len(self._cases):
            avg = sum(self._episode_rewards) / len(self._episode_rewards) if self._episode_rewards else SUBMISSION_SCORE_FLOOR
            avg = _submission_safe_score(avg)
            return ClinicalObservation(
                task_id=self._current_task,
                task_name=TASK_NAMES[self._current_task],
                task_description="Episode already complete. Call reset() to start a new task.",
                input_data=None,
                study_context="",
                feedback="No action graded — episode was already done.",
                case_number=len(self._cases),
                total_cases=len(self._cases),
                task_score=avg,
                done=True,
                reward=avg,
                episode_summary=self._build_episode_summary(),
                difficulty_breakdown=self._build_difficulty_breakdown(),
            )

        case = self._cases[self._current_case_idx]
        score, feedback, sub_scores = self._grade(action, case)
        self._last_sub_scores = sub_scores
        self._current_case_attempts += 1

        if score < 1.0 and self._current_case_attempts < self.MAX_ATTEMPTS_PER_CASE:
            obs = self._make_observation(
                case,
                feedback=feedback,
                reward=_submission_safe_score(score),
                done=False,
            )
            obs.case_attempt = self._current_case_attempts + 1
            return obs

        attempt_penalty = 1.0 if self._current_case_attempts == 1 else 0.85
        final_score = _submission_safe_score(score * attempt_penalty)
        self._episode_rewards.append(final_score)
        self._case_summaries.append(
            {
                "case_id": case.get("case_id", f"case_{self._current_case_idx + 1}"),
                "difficulty": case.get("difficulty", "unlabeled"),
                "score": final_score,
                "raw_score": round(score, 4),
                "attempts": self._current_case_attempts,
                "sub_scores": sub_scores,
            }
        )

        self._current_case_idx += 1
        self._current_case_attempts = 0
        done = self._current_case_idx >= len(self._cases)
        avg_reward = _submission_safe_score(sum(self._episode_rewards) / len(self._episode_rewards))

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
                task_score=avg_reward,
                done=True,
                reward=avg_reward,
                detection_score=float(sub_scores.get("detection_score", 0.0)),
                correction_score=float(sub_scores.get("correction_score", 0.0)),
                field_scores=sub_scores.get("field_scores", {}),
                sub_scores=sub_scores,
                episode_summary=self._build_episode_summary(),
                difficulty_breakdown=self._build_difficulty_breakdown(),
            )

        return self._make_observation(
            self._cases[self._current_case_idx],
            feedback=feedback,
            reward=final_score,
            done=False,
        )

    @property
    def state(self) -> State:
        return self._state

    def _make_observation(self, case: dict, feedback: str, reward: float, done: bool) -> ClinicalObservation:
        sub_scores = self._last_sub_scores or {}
        task_score = (
            _submission_safe_score(sum(self._episode_rewards) / len(self._episode_rewards))
            if self._episode_rewards
            else SUBMISSION_SCORE_FLOOR
        )
        return ClinicalObservation(
            task_id=self._current_task,
            task_name=TASK_NAMES[self._current_task],
            task_description=case["task_description"],
            input_data=case["input_data"],
            study_context=case["study_context"],
            feedback=feedback,
            case_number=self._current_case_idx + 1,
            total_cases=len(self._cases),
            task_score=task_score,
            case_attempt=self._current_case_attempts + 1,
            detection_score=float(sub_scores.get("detection_score", 0.0)),
            correction_score=float(sub_scores.get("correction_score", 0.0)),
            field_scores=sub_scores.get("field_scores", {}),
            sub_scores=sub_scores,
            done=done,
            reward=_submission_safe_score(reward),
        )

    def _build_episode_summary(self) -> dict:
        return {
            entry["case_id"]: {
                "score": entry["score"],
                "raw_score": entry["raw_score"],
                "attempts": entry["attempts"],
                "difficulty": entry["difficulty"],
                "sub_scores": entry["sub_scores"],
            }
            for entry in self._case_summaries
        }

    def _build_difficulty_breakdown(self) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for entry in self._case_summaries:
            buckets.setdefault(entry["difficulty"], []).append(entry["score"])
        return {difficulty: round(sum(scores) / len(scores), 4) for difficulty, scores in buckets.items()}

    def _grade(self, action: ClinicalAction, case: dict) -> tuple[float, str, dict]:
        gt = case["ground_truth"]
        output = action.output_data

        if self._current_task == 1:
            return grade_task1(output, gt)
        if self._current_task == 2:
            return grade_task2(output, gt)
        if self._current_task == 3:
            return grade_task3(output, gt)
        if self._current_task == 4:
            return grade_task4(output, gt)
        return 0.0, f"Unknown task id: {self._current_task}", {}
