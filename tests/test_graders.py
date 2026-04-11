"""Tests for all clinical data graders."""
import pytest
from graders.grader_task1 import grade_task1
from graders.grader_task2 import grade_task2
from graders.grader_task3 import grade_task3
from graders.grader_task4 import grade_task4


# ── Score range invariant ────────────────────────────────────────────────────

def assert_valid_score(score: float, context: str = ""):
    """All graders must return scores strictly between 0 and 1."""
    assert 0 < score < 1, f"Score {score} is not strictly in (0, 1). {context}"


# ── Task 1 ──────────────────────────────────────────────────────────────────

GT_T1 = {
    "USUBJID": "CDISCPILOT01-01-001",
    "AGE": 52,
    "SEX": "M",
    "RACE": "WHITE",
    "RFSTDTC": "2024-06-15",
    "COUNTRY": "USA",
}


def test_task1_perfect_score_clamped_to_max():
    score, feedback, sub = grade_task1(GT_T1, GT_T1)
    assert_valid_score(score)
    assert score == pytest.approx(0.99)
    assert sub["correction_score"] == pytest.approx(1.0)
    assert sub["correct_fields"] == 6


def test_task1_one_wrong_field():
    bad = {**GT_T1, "SEX": "F"}
    score, feedback, sub = grade_task1(bad, GT_T1)
    assert_valid_score(score)
    assert score < 0.99
    assert sub["correct_fields"] == 5


def test_task1_hallucinated_field_penalised():
    bad = {**GT_T1, "EXTRA": "hallucinated"}
    score, feedback, sub = grade_task1(bad, GT_T1)
    assert_valid_score(score)
    assert sub["hallucination_penalty"] > 0


def test_task1_wrong_type_returns_minimum():
    score, feedback, sub = grade_task1("not a dict", GT_T1)
    assert_valid_score(score)
    assert score == pytest.approx(0.01)


def test_task1_ambiguous_race_unknown():
    """Hard case: Hispanic maps to UNKNOWN, not a standard CDISC race term."""
    gt = {**GT_T1, "RACE": "UNKNOWN"}
    agent = {**GT_T1, "RACE": "UNKNOWN"}
    score, _, sub = grade_task1(agent, gt)
    assert_valid_score(score)
    assert score == pytest.approx(0.99)
    assert sub["field_scores"]["RACE"] == pytest.approx(1.0)


def test_task1_wrong_race_on_hard_case():
    """Agent guesses WHITE instead of UNKNOWN for Hispanic subject — should lose points."""
    gt = {**GT_T1, "RACE": "UNKNOWN"}
    agent = {**GT_T1, "RACE": "WHITE"}
    score, _, sub = grade_task1(agent, gt)
    assert_valid_score(score)
    assert score < 0.99
    assert sub["field_scores"]["RACE"] == 0.0


# ── Task 2 ──────────────────────────────────────────────────────────────────

GT_T2_WITH_VIOLATIONS = {
    "violations": [
        {"field": "AESTDTC", "issue": "Wrong format", "corrected_value": "2024-01-01"},
        {"field": "AESEV", "issue": "Wrong case", "corrected_value": "MILD"},
    ]
}
GT_T2_CLEAN = {"violations": []}


def test_task2_perfect_detection_and_correction():
    score, feedback, sub = grade_task2(GT_T2_WITH_VIOLATIONS, GT_T2_WITH_VIOLATIONS)
    assert_valid_score(score)
    assert score == pytest.approx(0.99)
    assert sub["detection_score"] == pytest.approx(1.0)
    assert sub["correction_score"] == pytest.approx(1.0)


def test_task2_clean_record_correctly_identified():
    score, feedback, sub = grade_task2(GT_T2_CLEAN, GT_T2_CLEAN)
    assert_valid_score(score)
    assert score == pytest.approx(0.99)
    assert sub["detection_score"] == pytest.approx(1.0)
    assert sub["false_positives"] == 0


def test_task2_false_positive_on_clean_record():
    agent = {"violations": [{"field": "AESEV", "issue": "Wrong", "corrected_value": "MILD"}]}
    score, feedback, sub = grade_task2(agent, GT_T2_CLEAN)
    assert_valid_score(score)
    assert score < 0.99
    assert sub["false_positives"] == 1


def test_task2_missed_violation():
    score, feedback, sub = grade_task2({"violations": []}, GT_T2_WITH_VIOLATIONS)
    assert_valid_score(score)
    assert sub["detection_score"] == pytest.approx(0.0)
    assert sub["missed"] == 2


def test_task2_wrong_type_returns_minimum():
    score, feedback, sub = grade_task2("bad", GT_T2_WITH_VIOLATIONS)
    assert_valid_score(score)
    assert score == pytest.approx(0.01)


# ── Task 3 ──────────────────────────────────────────────────────────────────

GT_T3 = [
    {"USUBJID": "01-001", "VISIT": "SCREENING 1", "PARAM": "HbA1c [mmol/mol]",
     "AVAL": 62.84175, "BASE": 62.84175, "CHG": 0.0, "PCHG": 0.0, "ABLFL": "Y", "ANL01FL": ""},
    {"USUBJID": "01-001", "VISIT": "WEEK 12", "PARAM": "HbA1c [mmol/mol]",
     "AVAL": 51.91275, "BASE": 62.84175, "CHG": -10.93, "PCHG": -17.39, "ABLFL": "", "ANL01FL": "Y"},
]


def test_task3_perfect_score():
    score, feedback, sub = grade_task3(GT_T3, GT_T3)
    assert_valid_score(score)
    assert score == pytest.approx(0.99)


def test_task3_aval_rounding_within_tolerance():
    """Agent rounds AVAL to 3dp — should pass with loose AVAL tolerance (0.001)."""
    agent = [{**r} for r in GT_T3]
    agent[0] = {**agent[0], "AVAL": 62.842}  # within 0.001 of 62.84175
    score, feedback, sub = grade_task3(agent, GT_T3)
    assert_valid_score(score)
    assert score > 0.9


def test_task3_wrong_pchg_fails():
    agent = [{**r} for r in GT_T3]
    agent[1] = {**agent[1], "PCHG": -25.0}  # clearly wrong (expected -17.39)
    score, feedback, sub = grade_task3(agent, GT_T3)
    assert_valid_score(score)
    assert score < 0.99


def test_task3_pchg_relative_tolerance():
    """PCHG within 0.5% relative tolerance should pass."""
    agent = [{**r} for r in GT_T3]
    agent[1] = {**agent[1], "PCHG": -17.40}  # within 0.5% of -17.39
    score, feedback, sub = grade_task3(agent, GT_T3)
    assert_valid_score(score)
    assert score == pytest.approx(0.99)


def test_task3_wrong_type_returns_minimum():
    score, feedback, sub = grade_task3("not a list", GT_T3)
    assert_valid_score(score)
    assert score == pytest.approx(0.01)


def test_task3_missing_visit():
    agent = [GT_T3[0]]  # only screening visit, missing WEEK 12
    score, feedback, sub = grade_task3(agent, GT_T3)
    assert_valid_score(score)
    assert score < 0.6


# ── Task 4 ──────────────────────────────────────────────────────────────────

GT_T4_ONE_ISSUE = {
    "issues": [
        {"type": "dm_ex_date_mismatch", "domain": "DM/EX",
         "field": "RFSTDTC/EXSTDTC", "description": "DM RFSTDTC 2024-03-15 does not match EX EXSTDTC 2024-03-20"}
    ]
}
GT_T4_TWO_ISSUES = {
    "issues": [
        {"type": "dm_ex_date_mismatch", "domain": "DM/EX", "field": "RFSTDTC/EXSTDTC", "description": "Date mismatch"},
        {"type": "orphan_sae", "domain": "AE/DS", "field": "AESER", "description": "SAE has no DS record"},
    ]
}
GT_T4_CLEAN = {"issues": []}


def test_task4_correct_issue_detected():
    score, feedback, sub = grade_task4(GT_T4_ONE_ISSUE, GT_T4_ONE_ISSUE)
    assert_valid_score(score)
    assert score > 0.5
    assert sub["detection_score"] == pytest.approx(1.0)


def test_task4_clean_record_correctly_identified():
    score, feedback, sub = grade_task4(GT_T4_CLEAN, GT_T4_CLEAN)
    assert_valid_score(score)
    assert score == pytest.approx(0.99)
    assert sub["detection_score"] == pytest.approx(1.0)


def test_task4_false_positive_on_clean_record():
    score, feedback, sub = grade_task4(GT_T4_ONE_ISSUE, GT_T4_CLEAN)
    assert_valid_score(score)
    assert score < 0.99


def test_task4_missed_issue():
    score, feedback, sub = grade_task4(GT_T4_CLEAN, GT_T4_ONE_ISSUE)
    assert_valid_score(score)
    assert score < 0.5
    assert sub["missed"] == 1


def test_task4_partial_detection_two_issues():
    agent = {"issues": [GT_T4_TWO_ISSUES["issues"][0]]}  # only finds first issue
    score, feedback, sub = grade_task4(agent, GT_T4_TWO_ISSUES)
    assert_valid_score(score)
    assert 0.01 < score < 0.99
    assert sub["missed"] == 1


def test_task4_alias_issue_type_still_gets_detection_credit():
    agent = {
        "issues": [
            {
                "type": "first dose date mismatch",
                "domain": "DM and EX",
                "field": "RFSTDTC and EXSTDTC",
                "description": "DM RFSTDTC does not match earliest EXSTDTC",
            }
        ]
    }
    score, _, sub = grade_task4(agent, GT_T4_ONE_ISSUE)
    assert_valid_score(score)
    assert sub["detection_score"] == pytest.approx(1.0)
    assert sub["correction_score"] > 0.5


def test_task4_orphan_sae_with_ae_only_domain_gets_partial_detail_credit():
    agent = {
        "issues": [
            {
                "type": "serious AE without DS",
                "domain": "AE",
                "field": "AETERM",
                "description": "Serious AE has no DS record",
            }
        ]
    }
    gt = {
        "issues": [
            {"type": "orphan_sae", "domain": "AE/DS", "field": "AESER", "description": "SAE has no DS record"}
        ]
    }
    score, _, sub = grade_task4(agent, gt)
    assert_valid_score(score)
    assert sub["detection_score"] == pytest.approx(1.0)
    assert sub["correction_score"] > 0.0


def test_task4_vague_inconsistency_does_not_get_detection_credit():
    """Vague 'inconsistency' label must NOT match any specific issue type."""
    agent = {"issues": [{"type": "inconsistency", "domain": "unknown", "field": "unknown", "description": "some issue"}]}
    score, _, sub = grade_task4(agent, GT_T4_ONE_ISSUE)
    assert_valid_score(score)
    assert sub["detection_score"] == pytest.approx(0.0)


def test_task4_wrong_type_returns_minimum():
    score, feedback, sub = grade_task4("bad input", GT_T4_ONE_ISSUE)
    assert_valid_score(score)
    assert score == pytest.approx(0.01)


# ── Score range invariant across all graders ─────────────────────────────────

@pytest.mark.parametrize("fn,agent,gt", [
    (grade_task1, GT_T1, GT_T1),
    (grade_task1, "bad", GT_T1),
    (grade_task1, {**GT_T1, "SEX": "F"}, GT_T1),
    (grade_task2, GT_T2_WITH_VIOLATIONS, GT_T2_WITH_VIOLATIONS),
    (grade_task2, GT_T2_CLEAN, GT_T2_CLEAN),
    (grade_task2, "bad", GT_T2_WITH_VIOLATIONS),
    (grade_task3, GT_T3, GT_T3),
    (grade_task3, "bad", GT_T3),
    (grade_task4, GT_T4_ONE_ISSUE, GT_T4_ONE_ISSUE),
    (grade_task4, GT_T4_CLEAN, GT_T4_CLEAN),
    (grade_task4, "bad", GT_T4_ONE_ISSUE),
])
def test_score_strictly_between_0_and_1(fn, agent, gt):
    score, _, __ = fn(agent, gt)
    assert 0 < score < 1, f"{fn.__name__}: score {score} is not strictly in (0, 1)"
