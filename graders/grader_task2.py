"""
Task 2 Grader: SDTM AE validation — violation detection + correction.

Score = 0.5 * F1(violation detection) + 0.5 * correction_accuracy
Score range: strictly (0.0, 1.0) — never exactly 0 or 1.
"""

from typing import Any


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0, 1)."""
    return max(0.01, min(0.99, score))


def grade_task2(agent_output: Any, ground_truth: dict) -> tuple[float, str]:
    """
    Grade agent output for Task 2 (SDTM validation).

    Args:
        agent_output: Agent's submitted JSON — must be dict with 'violations' list.
                      Each violation: {field, issue, corrected_value}
        ground_truth: Dict with 'violations' list of ground-truth violations.

    Returns:
        Tuple of (score 0.0–1.0, feedback string).
    """
    gt_violations: list[dict] = ground_truth.get("violations", [])

    # Normalise agent output
    if isinstance(agent_output, dict):
        agent_violations: list[dict] = agent_output.get("violations", [])
    elif isinstance(agent_output, list):
        agent_violations = agent_output
    else:
        return _clamp(0.0), f"Expected a JSON object with 'violations' key, got {type(agent_output).__name__}."

    gt_fields = {v["field"] for v in gt_violations}
    agent_fields = {v["field"] for v in agent_violations if isinstance(v, dict) and "field" in v}

    # Special case: ground truth has no violations
    if not gt_violations:
        if not agent_violations:
            return _clamp(1.0), "Score: 0.99. Correctly identified no violations in this record."
        fp_count = len(agent_violations)
        penalty = min(fp_count * 0.25, 1.0)
        score = _clamp(round(max(0.0, 1.0 - penalty), 4))
        fp_fields = [v.get("field", "?") for v in agent_violations if isinstance(v, dict)]
        return score, (
            f"Score: {score:.2f}. This record has NO violations — all fields conform to SDTM. "
            f"Agent incorrectly flagged {fp_count} false positive(s): {fp_fields}"
        )

    # F1 score on violation detection
    tp = len(gt_fields & agent_fields)
    precision = tp / len(agent_fields) if agent_fields else 0.0
    recall = tp / len(gt_fields) if gt_fields else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Correction accuracy
    gt_corrections = {v["field"]: str(v["corrected_value"]).strip() for v in gt_violations}
    agent_corrections = {
        v["field"]: str(v.get("corrected_value", "")).strip()
        for v in agent_violations
        if isinstance(v, dict) and "field" in v
    }
    correct_fixes = sum(
        1 for f in gt_fields
        if agent_corrections.get(f) == gt_corrections.get(f)
    )
    correction_score = correct_fixes / len(gt_fields) if gt_fields else 0.0

    score = _clamp(round(0.5 * f1 + 0.5 * correction_score, 4))

    # Build feedback
    missed = gt_fields - agent_fields
    false_positives = agent_fields - gt_fields
    wrong_corrections = [
        f for f in (gt_fields & agent_fields)
        if agent_corrections.get(f) != gt_corrections.get(f)
    ]

    lines = [f"Score: {score:.2f}  (F1={f1:.2f}, correction={correction_score:.2f})"]
    if missed:
        lines.append(f"Missed violations: {sorted(missed)}")
    if false_positives:
        lines.append(f"False positives: {sorted(false_positives)}")
    if wrong_corrections:
        for f in wrong_corrections:
            lines.append(
                f"  {f}: expected correction {gt_corrections[f]!r}, "
                f"got {agent_corrections.get(f)!r}"
            )

    return score, "\n".join(lines)
