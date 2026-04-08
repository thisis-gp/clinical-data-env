"""
Task 3 Grader: SDTM LB → ADAM ADLB derivation.

Scores numeric fields within tolerance and exact-match string fields.
Score range: 0.0 – 1.0
"""

from typing import Any


NUMERIC_FIELDS = ["AVAL", "BASE", "CHG", "PCHG"]
EXACT_FIELDS = ["USUBJID", "VISIT", "PARAM", "ABLFL", "ANL01FL"]
TOLERANCE = 0.01


def grade_task3(agent_records: Any, ground_truth_records: list) -> tuple[float, str]:
    """
    Grade agent output for Task 3 (SDTM → ADAM derivation).

    Args:
        agent_records: Agent's submitted list of ADAM records.
        ground_truth_records: List of correct ADAM records.

    Returns:
        Tuple of (score 0.0–1.0, feedback string).
    """
    if not isinstance(agent_records, list):
        return 0.0, f"Expected a JSON array, got {type(agent_records).__name__}."

    total = 0
    correct = 0
    feedback_lines = []

    for gt_rec in ground_truth_records:
        visit = gt_rec["VISIT"]
        agent_rec = next(
            (r for r in agent_records if isinstance(r, dict) and r.get("VISIT") == visit),
            {},
        )

        if not agent_rec:
            feedback_lines.append(f"  {visit}: record missing entirely")
            total += len(NUMERIC_FIELDS) + len(EXACT_FIELDS)
            continue

        for field in NUMERIC_FIELDS:
            total += 1
            try:
                agent_val = float(agent_rec.get(field, float("inf")))
                expected_val = float(gt_rec[field])
                if abs(agent_val - expected_val) <= TOLERANCE:
                    correct += 1
                else:
                    feedback_lines.append(
                        f"  {visit}.{field}: expected {expected_val}, got {agent_val}"
                    )
            except (TypeError, ValueError):
                feedback_lines.append(
                    f"  {visit}.{field}: expected {gt_rec[field]}, got non-numeric {agent_rec.get(field)!r}"
                )

        for field in EXACT_FIELDS:
            total += 1
            expected_val = str(gt_rec[field]).strip()
            agent_val = str(agent_rec.get(field, "")).strip()
            if agent_val == expected_val:
                correct += 1
            else:
                feedback_lines.append(
                    f"  {visit}.{field}: expected {expected_val!r}, got {agent_val!r}"
                )

    score = round(correct / total, 4) if total > 0 else 0.0

    if feedback_lines:
        feedback = f"Score: {score:.2f}. Issues:\n" + "\n".join(feedback_lines)
    else:
        feedback = f"Score: {score:.2f}. All fields correct."

    return score, feedback
