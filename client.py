"""Clinical Data Standardization Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import ClinicalAction, ClinicalObservation
except ImportError:
    from models import ClinicalAction, ClinicalObservation


class ClinicalDataEnv(EnvClient[ClinicalAction, ClinicalObservation, State]):
    """
    WebSocket client for the Clinical Data Standardization environment.

    Connects to a running OpenEnv server and provides reset/step/state methods.

    Example:
        >>> with ClinicalDataEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset()
        ...     print(result.observation.task_name)
        ...     action = ClinicalAction(
        ...         task_id=1,
        ...         output_data={"USUBJID": "CDISCPILOT01-01-001", ...},
        ...         reasoning="Mapped gender Male → M"
        ...     )
        ...     result = env.step(action)
        ...     print(result.reward)
    """

    def _step_payload(self, action: ClinicalAction) -> Dict:
        return {
            "task_id": action.task_id,
            "output_data": action.output_data,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ClinicalObservation]:
        obs_data = payload.get("observation", {})
        observation = ClinicalObservation(
            task_id=obs_data.get("task_id", 0),
            task_name=obs_data.get("task_name", ""),
            task_description=obs_data.get("task_description", ""),
            input_data=obs_data.get("input_data"),
            study_context=obs_data.get("study_context", ""),
            feedback=obs_data.get("feedback", ""),
            case_number=obs_data.get("case_number", 0),
            total_cases=obs_data.get("total_cases", 0),
            task_score=obs_data.get("task_score", 0.0),
            case_attempt=obs_data.get("case_attempt", 1),
            detection_score=obs_data.get("detection_score", 0.0),
            correction_score=obs_data.get("correction_score", 0.0),
            field_scores=obs_data.get("field_scores", {}),
            sub_scores=obs_data.get("sub_scores", {}),
            episode_summary=obs_data.get("episode_summary", {}),
            difficulty_breakdown=obs_data.get("difficulty_breakdown", {}),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
