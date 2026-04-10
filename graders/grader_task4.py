"""
Task 4 Grader: Cross-domain SDTM consistency validation.

Checks cross-domain inconsistencies across DM, EX, CM, AE, and DS domains.
"""

from typing import Any


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0, 1)."""
    return max(0.01, min(0.99, score))


def grade_task4(agent_output: Any, ground_truth: dict) -> tuple[float, str, dict]:
    """Grade agent output for Task 4 (cross-domain validation)."""
    if not isinstance(agent_output, dict):
        return _clamp(0.0), f"Expected a JSON object with 'issues' key, got {type(agent_output).__name__}.", {
            "detection_score": 0.0,
            "correction_score": 0.0,
            "field_scores": {},
        }

    agent_issues: list[dict] = agent_output.get("issues", [])
    if not isinstance(agent_issues, list):
        return _clamp(0.0), "Expected 'issues' to be a list.", {
            "detection_score": 0.0,
            "correction_score": 0.0,
            "field_scores": {},
        }

    gt_issues: list[dict] = ground_truth.get("issues", [])
    gt_types = {issue["type"] for issue in gt_issues}
    agent_types = {issue["type"] for issue in agent_issues if isinstance(issue, dict) and "type" in issue}

    if not gt_issues:
        if not agent_issues:
            return _clamp(1.0), "Score: 0.99. Correctly identified no cross-domain inconsistencies.", {
                "detection_score": 1.0,
                "correction_score": 1.0,
                "field_scores": {},
            }
        fp_count = len(agent_issues)
        score = _clamp(round(max(0.0, 1.0 - min(fp_count * 0.25, 1.0)), 4))
        return score, f"Score: {score:.2f}. Agent reported {fp_count} false cross-domain issue(s).", {
            "detection_score": 0.0,
            "correction_score": 0.0,
            "field_scores": {},
        }

    tp = len(gt_types & agent_types)
    precision = tp / len(agent_types) if agent_types else 0.0
    recall = tp / len(gt_types) if gt_types else 0.0
    detection_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    field_scores: dict[str, float] = {}
    detail_hits = 0.0
    for gt_issue in gt_issues:
        issue_type = gt_issue["type"]
        match = next((issue for issue in agent_issues if isinstance(issue, dict) and issue.get("type") == issue_type), None)
        if match is None:
            field_scores[issue_type] = 0.0
            continue
        gt_domain = str(gt_issue.get("domain", "")).upper()
        gt_field = str(gt_issue.get("field", "")).upper()
        agent_domain = str(match.get("domain", "")).upper()
        agent_field = str(match.get("field", "")).upper()
        agent_desc = str(match.get("description", "")).upper()

        issue_score = 0.0
        if gt_domain and (gt_domain in agent_domain or gt_domain in agent_desc):
            issue_score += 0.5
        if gt_field and (gt_field in agent_field or gt_field in agent_desc):
            issue_score += 0.5
        field_scores[issue_type] = issue_score
        detail_hits += issue_score

    correction_score = detail_hits / len(gt_issues) if gt_issues else 0.0
    score = _clamp(round(0.6 * detection_f1 + 0.4 * correction_score, 4))

    missed_types = gt_types - agent_types
    false_positive_types = agent_types - gt_types
    lines = [f"Score: {score:.2f}  (detection_f1={detection_f1:.2f}, correction={correction_score:.2f})"]
    if missed_types:
        lines.append(f"Missed issue types: {sorted(missed_types)}")
    if false_positive_types:
        lines.append(f"False positive issue types: {sorted(false_positive_types)}")

    return score, "\n".join(lines), {
        "detection_score": round(detection_f1, 4),
        "correction_score": round(correction_score, 4),
        "field_scores": field_scores,
        "missed": len(missed_types),
        "false_positives": len(false_positive_types),
    }
