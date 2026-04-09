from graders.grader_task1 import grade_task1
from graders.grader_task2 import grade_task2
from graders.grader_task3 import grade_task3
from graders.grader_task4 import grade_task4


def test_task1_ambiguous_mapping_scores_full_credit():
    ground_truth = {
        "USUBJID": "CT-ONC-201-007-1009",
        "AGE": "44",
        "SEX": "F",
        "RACE": "UNKNOWN",
        "RFSTDTC": "2025-08-10",
        "COUNTRY": "DEU",
    }
    agent_output = {
        "USUBJID": "CT-ONC-201-007-1009",
        "AGE": "44",
        "SEX": "F",
        "RACE": "UNKNOWN",
        "RFSTDTC": "2025-08-10",
        "COUNTRY": "DEU",
    }

    score, _, sub_scores = grade_task1(agent_output, ground_truth)

    assert score == 1.0
    assert sub_scores["detection_score"] == 1.0
    assert sub_scores["correction_score"] == 1.0
    assert sub_scores["field_scores"]["RACE"] == 1.0


def test_task2_zero_violation_case_rewards_empty_output():
    score, feedback, sub_scores = grade_task2({"violations": []}, {"violations": []})

    assert score == 1.0
    assert "no violations" in feedback.lower()
    assert sub_scores["detection_score"] == 1.0
    assert sub_scores["correction_score"] == 1.0


def test_task2_zero_violation_case_penalizes_false_positives():
    score, _, sub_scores = grade_task2(
        {"violations": [{"field": "AESER", "issue": "unexpected", "corrected_value": "N"}]},
        {"violations": []},
    )

    assert score == 0.75
    assert sub_scores["false_positives"] == 1
    assert sub_scores["detection_score"] == 0.0


def test_task3_pchg_uses_relative_tolerance():
    ground_truth = [
        {
            "USUBJID": "01-01-001",
            "VISIT": "WEEK 4",
            "PARAM": "ALT",
            "AVAL": 110.0,
            "BASE": 100.0,
            "CHG": 10.0,
            "PCHG": 10.0,
            "ABLFL": "",
            "ANL01FL": "Y",
        }
    ]
    agent_output = [
        {
            "USUBJID": "01-01-001",
            "VISIT": "WEEK 4",
            "PARAM": "ALT",
            "AVAL": 110.04,
            "BASE": 100.04,
            "CHG": 10.0,
            "PCHG": 10.009,
            "ABLFL": "",
            "ANL01FL": "Y",
        }
    ]

    score, _, sub_scores = grade_task3(agent_output, ground_truth)

    assert score == 1.0
    assert sub_scores["field_scores"]["WEEK 4.PCHG"] == 1.0
    assert sub_scores["field_scores"]["WEEK 4.AVAL"] == 1.0


def test_task4_detects_cross_domain_issue_and_scores_details():
    ground_truth = {
        "issues": [
            {
                "type": "orphan_sae",
                "domain": "AE/DS",
                "field": "AESER",
                "description": "Serious AE has no matching DS record.",
            }
        ]
    }
    agent_output = {
        "issues": [
            {
                "type": "orphan_sae",
                "domain": "AE/DS",
                "field": "AESER",
                "description": "AESER flagged serious event without DS evidence.",
            }
        ]
    }

    score, _, sub_scores = grade_task4(agent_output, ground_truth)

    assert score == 1.0
    assert sub_scores["detection_score"] == 1.0
    assert sub_scores["correction_score"] == 1.0
    assert sub_scores["field_scores"]["orphan_sae"] == 1.0
