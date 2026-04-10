"""
Task 3 Grader: SDTM LB -> ADAM ADLB derivation.

Uses field-specific tolerances:
- AVAL/BASE: slightly looser absolute tolerance to avoid harmless rounding penalties
- CHG: standard absolute tolerance
- PCHG: relative tolerance plus a small absolute floor
"""

from typing import Any


NUMERIC_FIELDS = ["AVAL", "BASE", "CHG", "PCHG"]
EXACT_FIELDS = ["USUBJID", "VISIT", "PARAM", "ABLFL", "ANL01FL"]
ABS_TOLERANCE = {
    "AVAL": 0.05,
    "BASE": 0.05,
    "CHG": 0.01,
}
PCHG_RELATIVE_TOLERANCE = 0.001  # 0.1%
PCHG_ABSOLUTE_FLOOR = 0.01


def _clamp(score: float) -> float:
    return max(0.01, min(0.99, score))


def _within_tolerance(field: str, agent_val: float, expected_val: float) -> bool:
    if field == "PCHG":
        tolerance = max(abs(expected_val) * PCHG_RELATIVE_TOLERANCE, PCHG_ABSOLUTE_FLOOR)
        return abs(agent_val - expected_val) <= tolerance
    return abs(agent_val - expected_val) <= ABS_TOLERANCE[field]


def grade_task3(agent_records: Any, ground_truth_records: list) -> tuple[float, str, dict]:
    """Grade agent output for Task 3 (SDTM -> ADAM derivation)."""
    if not isinstance(agent_records, list):
        return _clamp(0.0), f"Expected a JSON array, got {type(agent_records).__name__}.", {
            "detection_score": 0.0,
            "correction_score": 0.0,
            "field_scores": {},
        }

    total = 0
    correct = 0
    feedback_lines = []
    field_scores: dict[str, float] = {}
    record_hits = 0

    for gt_rec in ground_truth_records:
        visit = gt_rec["VISIT"]
        agent_rec = next((r for r in agent_records if isinstance(r, dict) and r.get("VISIT") == visit), {})

        if not agent_rec:
            feedback_lines.append(f"  {visit}: record missing entirely")
            for field in NUMERIC_FIELDS + EXACT_FIELDS:
                field_scores[f"{visit}.{field}"] = 0.0
            total += len(NUMERIC_FIELDS) + len(EXACT_FIELDS)
            continue

        record_hits += 1
        for field in NUMERIC_FIELDS:
            total += 1
            try:
                agent_val = float(agent_rec.get(field, float("inf")))
                expected_val = float(gt_rec[field])
                is_correct = _within_tolerance(field, agent_val, expected_val)
                field_scores[f"{visit}.{field}"] = 1.0 if is_correct else 0.0
                if is_correct:
                    correct += 1
                else:
                    feedback_lines.append(f"  {visit}.{field}: expected {expected_val}, got {agent_val}")
            except (TypeError, ValueError):
                field_scores[f"{visit}.{field}"] = 0.0
                feedback_lines.append(
                    f"  {visit}.{field}: expected {gt_rec[field]}, got non-numeric {agent_rec.get(field)!r}"
                )

        for field in EXACT_FIELDS:
            total += 1
            expected_val = str(gt_rec[field]).strip()
            agent_val = str(agent_rec.get(field, "")).strip()
            is_correct = agent_val == expected_val
            field_scores[f"{visit}.{field}"] = 1.0 if is_correct else 0.0
            if is_correct:
                correct += 1
            else:
                feedback_lines.append(f"  {visit}.{field}: expected {expected_val!r}, got {agent_val!r}")

    score = _clamp(round(correct / total, 4) if total > 0 else 0.0)
    detection_score = record_hits / len(ground_truth_records) if ground_truth_records else 0.0
    correction_score = score

    if feedback_lines:
        feedback = f"Score: {score:.2f}. Issues:\n" + "\n".join(feedback_lines)
    else:
        feedback = f"Score: {score:.2f}. All fields correct."

    return score, feedback, {
        "detection_score": round(detection_score, 4),
        "correction_score": round(correction_score, 4),
        "field_scores": field_scores,
        "correct_fields": correct,
        "total_fields": total,
    }
