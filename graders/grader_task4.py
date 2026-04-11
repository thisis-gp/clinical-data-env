"""
Task 4 Grader: Cross-domain SDTM consistency validation.

Checks cross-domain inconsistencies across DM, EX, CM, AE, and DS domains.
Uses canonical issue labels but also tolerates common near-miss phrasing so
the grader measures reasoning quality, not only literal string matching.
"""

from typing import Any


TYPE_ALIASES = {
    "dm_ex_date_mismatch": {
        "dm_ex_date_mismatch",
        "dm/ex date mismatch",
        "dm ex date mismatch",
        "dm rfstdtc mismatch",
        "first dose date mismatch",
        "rfstdtc exstdtc mismatch",
        "rfstdtc mismatch",
        "exstdtc mismatch",
        "first dose date inconsistency",
    },
    "prohibited_cm_before_first_dose": {
        "prohibited_cm_before_first_dose",
        "prohibited medication start date",
        "prohibited cm before first dose",
        "prohibited medication before first dose",
        "cm before first dose",
        "prohibited concomitant medication before first dose",
        "concomitant medication before first dose",
        "prohibited cm timing",
    },
    "orphan_sae": {
        "orphan_sae",
        "orphan sae",
        "serious ae without ds",
        "sae without ds",
        "ae ds inconsistency",
        "serious adverse event missing ds",
        "sae missing disposition",
        "serious ae no disposition",
    },
}

DOMAIN_ALIASES = {
    "DM/EX": {"DM/EX", "DM AND EX", "DM VS EX", "DM-EX"},
    "CM": {"CM", "CM/EX", "CM VS EX", "CM-EX"},
    "AE/DS": {"AE/DS", "AE AND DS", "AE VS DS", "AE-DS", "AE", "DS"},
}

FIELD_ALIASES = {
    "RFSTDTC/EXSTDTC": {"RFSTDTC/EXSTDTC", "RFSTDTC AND EXSTDTC", "RFSTDTC VS EXSTDTC", "RFSTDTC-EXSTDTC"},
    "CMSTDTC": {"CMSTDTC", "CMSTDTC/EXSTDTC", "CMSTDTC VS EXSTDTC", "CMSTDTC-EXSTDTC"},
    "AESER": {"AESER", "AETERM", "DSDECOD", "DSTERM"},
}


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0, 1)."""
    return max(0.01, min(0.99, score))


def _normalize(text: str) -> str:
    normalized = str(text).strip().upper()
    for old, new in {"/": " ", "-": " ", "_": " "}.items():
        normalized = normalized.replace(old, new)
    return " ".join(normalized.split())


def _canonical_issue_type(issue_type: str, description: str = "") -> str:
    combined = f"{issue_type} {description}"
    normalized = _normalize(combined)
    for canonical, aliases in TYPE_ALIASES.items():
        if any(_normalize(alias) in normalized for alias in aliases):
            return canonical
    return _normalize(issue_type).lower().replace(" ", "_")


def _matches_domain(expected: str, actual_domain: str, actual_desc: str) -> bool:
    expected_norm = _normalize(expected)
    actual_text = f"{actual_domain} {actual_desc}"
    actual_norm = _normalize(actual_text)
    aliases = DOMAIN_ALIASES.get(expected.upper(), {expected})
    return any(_normalize(alias) in actual_norm for alias in aliases)


def _matches_field(expected: str, actual_field: str, actual_desc: str) -> bool:
    expected_norm = _normalize(expected)
    actual_text = f"{actual_field} {actual_desc}"
    actual_norm = _normalize(actual_text)
    aliases = FIELD_ALIASES.get(expected.upper(), {expected})
    if any(_normalize(alias) in actual_norm for alias in aliases):
        return True
    # Fallback: reward partial overlap for slash-delimited compound fields.
    components = [component for component in expected_norm.split() if component]
    return any(component in actual_norm for component in components)


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

    canonical_gt_types = {
        _canonical_issue_type(issue["type"], issue.get("description", "")) for issue in gt_issues
    }
    canonical_agent_types = {
        _canonical_issue_type(issue.get("type", ""), issue.get("description", ""))
        for issue in agent_issues
        if isinstance(issue, dict) and issue.get("type")
    }

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

    tp = len(canonical_gt_types & canonical_agent_types)
    precision = tp / len(canonical_agent_types) if canonical_agent_types else 0.0
    recall = tp / len(canonical_gt_types) if canonical_gt_types else 0.0
    detection_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    field_scores: dict[str, float] = {}
    detail_hits = 0.0
    for gt_issue in gt_issues:
        canonical_type = _canonical_issue_type(gt_issue["type"], gt_issue.get("description", ""))
        match = next(
            (
                issue for issue in agent_issues
                if isinstance(issue, dict)
                and _canonical_issue_type(issue.get("type", ""), issue.get("description", "")) == canonical_type
            ),
            None,
        )
        if match is None:
            field_scores[canonical_type] = 0.0
            continue

        gt_domain = str(gt_issue.get("domain", ""))
        gt_field = str(gt_issue.get("field", ""))
        agent_domain = str(match.get("domain", ""))
        agent_field = str(match.get("field", ""))
        agent_desc = str(match.get("description", ""))

        issue_score = 0.0
        if gt_domain and _matches_domain(gt_domain, agent_domain, agent_desc):
            issue_score += 0.5
        if gt_field and _matches_field(gt_field, agent_field, agent_desc):
            issue_score += 0.5
        field_scores[canonical_type] = issue_score
        detail_hits += issue_score

    correction_score = detail_hits / len(gt_issues) if gt_issues else 0.0
    score = _clamp(round(0.6 * detection_f1 + 0.4 * correction_score, 4))

    missed_types = canonical_gt_types - canonical_agent_types
    false_positive_types = canonical_agent_types - canonical_gt_types
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
