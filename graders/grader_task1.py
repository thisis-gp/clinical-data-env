"""
Task 1 Grader: Raw EDC → SDTM DM mapping.

Scores field-by-field match against ground truth.
Penalises hallucinated extra fields not in the spec.
Score range: 0.0 – 1.0
"""

from typing import Any


EXPECTED_FIELDS = {"USUBJID", "AGE", "SEX", "RACE", "RFSTDTC", "COUNTRY"}


def grade_task1(agent_output: Any, ground_truth: dict) -> tuple[float, str]:
    """
    Grade agent output for Task 1 (EDC → SDTM mapping).

    Args:
        agent_output: Agent's submitted JSON (should be a dict).
        ground_truth: Correct SDTM DM record.

    Returns:
        Tuple of (score 0.0–1.0, feedback string).
    """
    if not isinstance(agent_output, dict):
        return 0.0, f"Expected a JSON object, got {type(agent_output).__name__}."

    correct = 0
    total = len(ground_truth)
    feedback_lines = []

    for field, expected in ground_truth.items():
        agent_val = agent_output.get(field)
        # Normalise: compare as strings, strip whitespace
        if str(agent_val).strip() == str(expected).strip():
            correct += 1
        else:
            feedback_lines.append(
                f"  {field}: expected {expected!r}, got {agent_val!r}"
            )

    # Penalise hallucinated fields (fields not in the spec)
    extra_fields = set(agent_output.keys()) - EXPECTED_FIELDS
    penalty = min(len(extra_fields) * 0.05, 0.2)  # cap penalty at 0.2

    base_score = correct / total if total > 0 else 0.0
    score = max(0.0, round(base_score - penalty, 4))

    if feedback_lines:
        feedback = f"Score: {score:.2f}. Incorrect fields:\n" + "\n".join(feedback_lines)
    else:
        feedback = f"Score: {score:.2f}. All fields correct."

    if extra_fields:
        feedback += f"\nHallucinated fields (not in spec): {sorted(extra_fields)}"

    return score, feedback
