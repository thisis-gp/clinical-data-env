"""
Pydantic models for the Clinical Data Standardization Environment.

Defines the Action and Observation types used by the OpenEnv server.
"""

from typing import Any, Union

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class ClinicalAction(Action):
    """Action submitted by the agent for a clinical data task."""

    task_id: int = Field(..., description="Task identifier: 1=EDC->SDTM, 2=Validation, 3=SDTM->ADAM, 4=Cross-domain")
    output_data: Union[dict, list] = Field(
        ...,
        description=(
            "Task 1: dict of SDTM field->value mappings. "
            "Task 2: dict with 'violations' list (each has field, issue, corrected_value). "
            "Task 3: list of ADAM records (one per visit). "
            "Task 4: dict with 'issues' list (each has type, domain, field, description)."
        ),
    )
    reasoning: str = Field(
        default="",
        description="Agent's explanation of its answer (not graded, used for logging).",
    )


class ClinicalObservation(Observation):
    """Observation returned by the environment after reset() or step()."""

    task_id: int = Field(default=0, description="Current task: 1, 2, 3, or 4")
    task_name: str = Field(default="", description="Human-readable task name")
    task_description: str = Field(default="", description="What the agent must do for this case")
    input_data: Any = Field(default=None, description="Raw input data for this case")
    study_context: str = Field(default="", description="CDISC rules and protocol instructions")
    feedback: str = Field(default="", description="Grader feedback from the previous step")
    case_number: int = Field(default=0, description="Current case index (1-based)")
    total_cases: int = Field(default=0, description="Total cases in this task")
    task_score: float = Field(default=0.0, description="Running average score for this task")
    case_attempt: int = Field(default=1, description="Current attempt number for this case (1=first, 2=retry)")
    detection_score: float = Field(default=0.0, description="Detection or structure-finding component score from the previous step")
    correction_score: float = Field(default=0.0, description="Correction or value-accuracy component score from the previous step")
    field_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-field or per-issue score breakdown for the previous step",
    )
    sub_scores: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional component score breakdown from the previous step",
    )
    episode_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-case summary when done=True",
    )
    difficulty_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Average score by difficulty bucket when done=True",
    )
