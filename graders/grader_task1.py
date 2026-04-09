"""
Task 1 Grader: Raw EDC -> SDTM DM mapping.

Scores field-by-field match against ground truth and penalizes hallucinated
extra fields not in the expected SDTM DM output.
"""

from typing import Any


EXPECTED_FIELDS = {"USUBJID", "AGE", "SEX", "RACE", "RFSTDTC", "COUNTRY"}


def grade_task1(agent_output: Any, ground_truth: dict) -> tuple[float, str, dict]:
    """Grade agent output for Task 1 (EDC -> SDTM mapping)."""
    if not isinstance(agent_output, dict):
        return 0.0, f"Expected a JSON object, got {type(agent_output).__name__}.", {
            "detection_score": 0.0,
            "correction_score": 0.0,
            "field_scores": {},
            "hallucination_penalty": 0.0,
        }

    correct = 0
    total = len(ground_truth)
    feedback_lines = []
    field_scores: dict[str, float] = {}

    for field, expected in ground_truth.items():
        agent_val = agent_output.get(field)
        is_correct = str(agent_val).strip() == str(expected).strip()
        field_scores[field] = 1.0 if is_correct else 0.0
        if is_correct:
            correct += 1
        else:
            feedback_lines.append(f"  {field}: expected {expected!r}, got {agent_val!r}")

    extra_fields = set(agent_output.keys()) - EXPECTED_FIELDS
    penalty = min(len(extra_fields) * 0.05, 0.2)
    detection_score = 1.0 if set(ground_truth.keys()).issubset(agent_output.keys()) else correct / total if total else 0.0
    correction_score = correct / total if total > 0 else 0.0
    score = max(0.0, round(correction_score - penalty, 4))

    if feedback_lines:
        feedback = f"Score: {score:.2f}. Incorrect fields:\n" + "\n".join(feedback_lines)
    else:
        feedback = f"Score: {score:.2f}. All fields correct."

    if extra_fields:
        feedback += f"\nHallucinated fields (not in spec): {sorted(extra_fields)}"

    return score, feedback, {
        "detection_score": round(detection_score, 4),
        "correction_score": round(correction_score, 4),
        "field_scores": field_scores,
        "hallucination_penalty": round(penalty, 4),
        "correct_fields": correct,
        "total_fields": total,
    }
