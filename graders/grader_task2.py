"""
Task 2 Grader: SDTM AE validation.

Score = 0.5 * detection F1 + 0.5 * correction accuracy.
Supports both violated and zero-violation records.
"""

from typing import Any


def grade_task2(agent_output: Any, ground_truth: dict) -> tuple[float, str, dict]:
    """Grade agent output for Task 2 (SDTM validation)."""
    gt_violations: list[dict] = ground_truth.get("violations", [])

    if isinstance(agent_output, dict):
        agent_violations: list[dict] = agent_output.get("violations", [])
    elif isinstance(agent_output, list):
        agent_violations = agent_output
    else:
        return 0.0, f"Expected a JSON object with 'violations' key, got {type(agent_output).__name__}.", {
            "detection_score": 0.0,
            "correction_score": 0.0,
            "field_scores": {},
        }

    gt_fields = {v["field"] for v in gt_violations}
    agent_fields = {v["field"] for v in agent_violations if isinstance(v, dict) and "field" in v}

    if not gt_violations:
        if not agent_violations:
            return 1.0, "Score: 1.00. Correctly identified no violations in this record.", {
                "detection_score": 1.0,
                "correction_score": 1.0,
                "field_scores": {},
                "false_positives": 0,
                "missed": 0,
            }

        fp_count = len(agent_violations)
        penalty = min(fp_count * 0.25, 1.0)
        score = round(max(0.0, 1.0 - penalty), 4)
        fp_fields = [v.get("field", "?") for v in agent_violations if isinstance(v, dict)]
        return score, (
            f"Score: {score:.2f}. This record has NO violations. "
            f"Agent incorrectly flagged {fp_count} false positive(s): {fp_fields}"
        ), {
            "detection_score": 0.0,
            "correction_score": 0.0,
            "field_scores": {field: 0.0 for field in fp_fields},
            "false_positives": fp_count,
            "missed": 0,
        }

    tp = len(gt_fields & agent_fields)
    precision = tp / len(agent_fields) if agent_fields else 0.0
    recall = tp / len(gt_fields) if gt_fields else 0.0
    detection_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    gt_corrections = {v["field"]: str(v["corrected_value"]).strip() for v in gt_violations}
    agent_corrections = {
        v["field"]: str(v.get("corrected_value", "")).strip()
        for v in agent_violations
        if isinstance(v, dict) and "field" in v
    }
    correct_fixes = sum(1 for field in gt_fields if agent_corrections.get(field) == gt_corrections.get(field))
    correction_score = correct_fixes / len(gt_fields) if gt_fields else 0.0
    score = round(0.5 * detection_f1 + 0.5 * correction_score, 4)

    missed = gt_fields - agent_fields
    false_positives = agent_fields - gt_fields
    wrong_corrections = [
        field for field in (gt_fields & agent_fields)
        if agent_corrections.get(field) != gt_corrections.get(field)
    ]
    field_scores = {
        field: (1.0 if agent_corrections.get(field) == gt_corrections.get(field) else 0.0)
        for field in gt_fields
    }

    lines = [f"Score: {score:.2f}  (F1={detection_f1:.2f}, correction={correction_score:.2f})"]
    if missed:
        lines.append(f"Missed violations: {sorted(missed)}")
    if false_positives:
        lines.append(f"False positives: {sorted(false_positives)}")
    if wrong_corrections:
        for field in wrong_corrections:
            lines.append(
                f"  {field}: expected correction {gt_corrections[field]!r}, got {agent_corrections.get(field)!r}"
            )

    return score, "\n".join(lines), {
        "detection_score": round(detection_f1, 4),
        "correction_score": round(correction_score, 4),
        "field_scores": field_scores,
        "false_positives": len(false_positives),
        "missed": len(missed),
        "wrong_corrections": len(wrong_corrections),
    }
