"""Generate a second, pharmaverse-derived benchmark set.

This keeps the existing toy cases intact and writes a more realistic set to:

    data/pharmaverse/task1_cases.json   (10 cases, race variety)
    data/pharmaverse/task2_cases.json   (10 cases, varied violation count)
    data/pharmaverse/task3_cases.json   (10 cases, increasing visit count)

Difficulty progression:
  Task 1: cases 1-5 all-USA/WHITE, cases 6-10 mix ASIAN/BLACK/AMERICAN INDIAN
  Task 2: cases 1-5 all 5 violations, cases 6-10 only 2-4 violations (harder to spot)
  Task 3: cases 1-3 three visits, 4-6 four visits, 7-9 five visits, 10 all 9 visits
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_REPO = ROOT / "pharmaverseraw"
SDTM_REPO = ROOT / "pharmaversesdtm"
OUT_DIR = ROOT / "data" / "pharmaverse"

COUNTRY_NAME = {
    "USA": "United States",
    "GBR": "United Kingdom",
    "DEU": "Germany",
    "IND": "India",
}

SEX_LABEL = {"M": "Male", "F": "Female"}

VISIT_ORDER = [
    "SCREENING 1",
    "WEEK 2",
    "WEEK 6",
    "WEEK 8",
    "WEEK 12",
    "WEEK 16",
    "WEEK 20",
    "WEEK 24",
    "WEEK 26",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def iso_to_ddmmyyyy(value: str) -> str:
    year, month, day = value.split("-")
    return f"{day}/{month}/{year}"


def title_case_phrase(value: str) -> str:
    return " ".join(part.capitalize() for part in value.lower().split(" "))


# ---------------------------------------------------------------------------
# Task 1 — EDC to SDTM DM (10 cases, race variety)
# ---------------------------------------------------------------------------

def build_task1_cases(dm_rows: list[dict[str, str]]) -> dict:
    # Prefer race variety: pick rows ensuring each race appears at least twice
    valid = [r for r in dm_rows if r["RFSTDTC"] and r["RFSTDTC"].count("-") == 2]
    by_race: dict[str, list] = defaultdict(list)
    for r in valid:
        by_race[r["RACE"]].append(r)

    selected: list[dict] = []
    # Round-robin across races to guarantee diversity
    race_order = ["WHITE", "BLACK OR AFRICAN AMERICAN", "ASIAN", "AMERICAN INDIAN OR ALASKA NATIVE"]
    iters = {race: iter(rows) for race, rows in by_race.items()}
    while len(selected) < 10:
        added = False
        for race in race_order:
            if len(selected) >= 10:
                break
            it = iters.get(race)
            if it is None:
                continue
            row = next(it, None)
            if row:
                selected.append(row)
                added = True
        if not added:
            break

    cases: list[dict] = []
    for index, row in enumerate(selected, start=1):
        usubjid_prefix = row["USUBJID"].split("-")[0]
        raw_input = {
            "STUDY": row["STUDYID"],
            "USUBJID_PREFIX": usubjid_prefix,
            "PATNUM": row["SUBJID"],
            "SITEID_RAW": row["SITEID"],
            "IT.AGE": row["AGE"],
            "IT.SEX": SEX_LABEL.get(row["SEX"], row["SEX"]),
            "IT.RACE": title_case_phrase(row["RACE"]),
            "IC_DT": iso_to_ddmmyyyy(row["RFSTDTC"]),
            "COUNTRY_NAME": COUNTRY_NAME.get(row["COUNTRY"], row["COUNTRY"]),
        }
        ground_truth = {
            "USUBJID": row["USUBJID"],
            "AGE": int(row["AGE"]),
            "SEX": row["SEX"],
            "RACE": row["RACE"],
            "RFSTDTC": row["RFSTDTC"],
            "COUNTRY": row["COUNTRY"],
        }
        cases.append(
            {
                "case_id": f"PV_T1_{index}",
                "task_description": (
                    "Convert this raw demographic record into an SDTM DM-style output "
                    "with exactly these fields: USUBJID, AGE, SEX, RACE, RFSTDTC, COUNTRY. "
                    "Return only those six fields — no extras."
                ),
                "study_context": (
                    "Map the raw demographic fields to the SDTM DM target variables.\n"
                    "- For this study, USUBJID uses the provided USUBJID_PREFIX, not the full STUDY text.\n"
                    "- USUBJID = USUBJID_PREFIX + '-' + SITEID_RAW + '-' + PATNUM\n"
                    "- AGE = integer years\n"
                    "- SEX = 'M' or 'F'  (Male -> M, Female -> F)\n"
                    "- RACE = uppercase SDTM controlled terminology\n"
                    "  Examples: 'White' -> 'WHITE', 'Black Or African American' -> 'BLACK OR AFRICAN AMERICAN',\n"
                    "  'Asian' -> 'ASIAN', 'American Indian Or Alaska Native' -> 'AMERICAN INDIAN OR ALASKA NATIVE'\n"
                    "- IC_DT is in DD/MM/YYYY format — day first, then month, then year\n"
                    "- RFSTDTC = ISO 8601 date YYYY-MM-DD converted from IC_DT\n"
                    "- COUNTRY = ISO 3166 alpha-3 code (e.g. 'United States' -> 'USA')"
                ),
                "input_data": raw_input,
                "ground_truth": ground_truth,
            }
        )

    return {
        "task_name": "task1_edc_to_sdtm",
        "description": "Pharmaverse-derived raw demographics mapped to SDTM DM outputs (10 cases, diverse races).",
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Task 2 — SDTM Validation (10 cases, varied violation count)
# ---------------------------------------------------------------------------

def build_task2_cases(ae_rows: list[dict[str, str]]) -> dict:
    # Collect 10 unique subject+term rows
    selected: list[dict[str, str]] = []
    seen_terms: set[tuple[str, str]] = set()
    for row in ae_rows:
        key = (row["USUBJID"], row["AETERM"])
        if key in seen_terms:
            continue
        seen_terms.add(key)
        selected.append(row)
        if len(selected) == 10:
            break

    severity_swap = {"MILD": "Mild", "MODERATE": "Moderate", "SEVERE": "Severe"}
    ser_swap = {"Y": "Yes", "N": "No"}
    rel_swap = {
        "PROBABLE": "Probably Related",
        "POSSIBLE": "Possibly Related",
        "REMOTE": "Remote",
        "NONE": "Not Related",
    }
    out_swap = {
        "RECOVERED/RESOLVED": "Resolved",
        "NOT RECOVERED/NOT RESOLVED": "Not Recovered",
        "FATAL": "Fatal",  # FATAL is already valid SDTM - leave as-is for harder cases
    }

    # Violation pattern per case index (0-based):
    # True = inject violation, False = leave field correct
    # Fields: [AESTDTC, AESEV, AESER, AEREL, AEOUT]
    violation_patterns = [
        [True,  True,  True,  True,  True ],  # case 1: all 5 wrong (easy)
        [True,  True,  True,  True,  True ],  # case 2: all 5 wrong
        [True,  True,  True,  True,  True ],  # case 3: all 5 wrong
        [True,  True,  True,  True,  True ],  # case 4: all 5 wrong
        [True,  True,  True,  True,  True ],  # case 5: all 5 wrong
        [True,  False, True,  True,  True ],  # case 6: AESEV already correct (4 violations)
        [True,  True,  False, True,  True ],  # case 7: AESER already correct (4 violations)
        [True,  True,  True,  False, True ],  # case 8: AEREL already correct - NA means not applicable (4 violations)
        [True,  False, True,  True,  False],  # case 9: AESEV + AEOUT already correct (3 violations)
        [True,  False, False, True,  True ],  # case 10: AESEV + AESER already correct (3 violations)
    ]

    cases: list[dict] = []
    for index, row in enumerate(selected, start=1):
        pattern = violation_patterns[index - 1]
        inject_date, inject_sev, inject_ser, inject_rel, inject_out = pattern

        bad_row: dict = {
            "USUBJID": row["USUBJID"],
            "AETERM": row["AETERM"],
            "AEDECOD": row["AEDECOD"],
        }

        violations: list[dict] = []

        # AESTDTC — always inject (date format violation is universal)
        if inject_date:
            bad_row["AESTDTC"] = f"{row['AESTDTC'][5:7]}-{row['AESTDTC'][8:10]}-{row['AESTDTC'][0:4]}"
            violations.append({
                "field": "AESTDTC",
                "issue": "Date not in ISO 8601 format YYYY-MM-DD",
                "corrected_value": row["AESTDTC"],
            })
        else:
            bad_row["AESTDTC"] = row["AESTDTC"]

        bad_row["AEENDTC"] = row["AEENDTC"]

        # AESEV
        if inject_sev:
            bad_row["AESEV"] = severity_swap.get(row["AESEV"], row["AESEV"].title())
            violations.append({
                "field": "AESEV",
                "issue": "Severity not in controlled terminology (must be MILD, MODERATE, or SEVERE)",
                "corrected_value": row["AESEV"],
            })
        else:
            bad_row["AESEV"] = row["AESEV"]  # already correct

        # AESER
        if inject_ser:
            bad_row["AESER"] = ser_swap.get(row["AESER"], row["AESER"])
            violations.append({
                "field": "AESER",
                "issue": "Serious event flag must be Y or N",
                "corrected_value": row["AESER"],
            })
        else:
            bad_row["AESER"] = row["AESER"]  # already correct

        # AEREL — skip if AEREL is NA (not applicable, leave correct)
        aerel_raw = row["AEREL"]
        if inject_rel and aerel_raw in rel_swap:
            bad_row["AEREL"] = rel_swap[aerel_raw]
            violations.append({
                "field": "AEREL",
                "issue": "Relationship value not in expected sponsor terminology",
                "corrected_value": aerel_raw,
            })
        else:
            bad_row["AEREL"] = aerel_raw  # NA or already correct

        # AEOUT
        aeout_raw = row["AEOUT"]
        if inject_out and aeout_raw in out_swap and aeout_raw != "FATAL":
            bad_row["AEOUT"] = out_swap[aeout_raw]
            violations.append({
                "field": "AEOUT",
                "issue": "Outcome value not in controlled terminology",
                "corrected_value": aeout_raw,
            })
        else:
            bad_row["AEOUT"] = aeout_raw  # already correct (FATAL or explicit keep)

        cases.append(
            {
                "case_id": f"PV_T2_{index}",
                "task_description": (
                    "Identify every terminology or formatting violation in this SDTM AE record. "
                    "Return a JSON object with key 'violations'. "
                    "Not every field is wrong — only report actual violations."
                ),
                "study_context": (
                    "Validate this AE record against expected SDTM conventions.\n"
                    "- AESTDTC must be YYYY-MM-DD (ISO 8601)\n"
                    "- AESEV must be one of: MILD, MODERATE, SEVERE (uppercase)\n"
                    "- AESER must be Y or N\n"
                    "- AEREL must use sponsor terminology: PROBABLE, POSSIBLE, REMOTE, NONE, or NA\n"
                    "- AEOUT must use exact SDTM term: RECOVERED/RESOLVED, NOT RECOVERED/NOT RESOLVED, or FATAL\n"
                    "Fields that already conform are NOT violations. Only report fields that need correction."
                ),
                "input_data": bad_row,
                "ground_truth": {"violations": violations},
            }
        )

    return {
        "task_name": "task2_sdtm_validation",
        "description": "Pharmaverse AE records with varied violation counts (2-5 per case).",
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Task 3 — SDTM to ADaM ADLB (10 cases, increasing visit count)
# ---------------------------------------------------------------------------

# 10 case definitions: (subject_index, visit_subset)
# subject_index 0/1/2 maps to the 3 available pharmaverse subjects
TASK3_CASES = [
    # Easy: 3 visits
    (0, ["SCREENING 1", "WEEK 12", "WEEK 24"]),
    (1, ["SCREENING 1", "WEEK 12", "WEEK 24"]),
    (2, ["SCREENING 1", "WEEK 12", "WEEK 24"]),
    # Medium: 4 visits
    (0, ["SCREENING 1", "WEEK 2", "WEEK 12", "WEEK 24"]),
    (1, ["SCREENING 1", "WEEK 6", "WEEK 12", "WEEK 24"]),
    (2, ["SCREENING 1", "WEEK 2", "WEEK 8", "WEEK 24"]),
    # Hard: 5 visits
    (0, ["SCREENING 1", "WEEK 2", "WEEK 8", "WEEK 16", "WEEK 24"]),
    (1, ["SCREENING 1", "WEEK 6", "WEEK 12", "WEEK 20", "WEEK 26"]),
    (2, ["SCREENING 1", "WEEK 2", "WEEK 8", "WEEK 16", "WEEK 26"]),
    # Very hard: all 9 visits
    (0, ["SCREENING 1", "WEEK 2", "WEEK 6", "WEEK 8", "WEEK 12", "WEEK 16", "WEEK 20", "WEEK 24", "WEEK 26"]),
]

TASK3_STUDY_CONTEXT = (
    "Derive one ADLB-style record per visit. "
    "Your output MUST be a JSON array where EVERY row contains ALL of these fields:\n"
    "  USUBJID, VISIT, PARAM, AVAL, BASE, CHG, PCHG, ABLFL, ANL01FL\n\n"
    "Derivation rules:\n"
    "- USUBJID: copy from input row\n"
    "- VISIT: copy from input row\n"
    "- PARAM: LBTEST + ' [' + LBSTRESU + ']'\n"
    "- AVAL: LBSTRESN (copy exactly, preserve all decimal places)\n"
    "- BASE: AVAL from the SCREENING 1 row (same value repeated on every row)\n"
    "- CHG: AVAL - BASE, rounded to 2 decimal places\n"
    "- PCHG: ((AVAL - BASE) / BASE) * 100, rounded to 2 decimal places\n"
    "- ABLFL: 'Y' for the SCREENING 1 row only; empty string '' for all other rows\n"
    "- ANL01FL: '' for SCREENING 1; 'Y' for every post-baseline visit\n\n"
    "Do NOT omit any field. Do NOT use null or 'N'. Use empty string '' where a flag is not set."
)


def build_task3_cases(lb_rows: list[dict[str, str]]) -> dict:
    hba1c = [
        r for r in lb_rows
        if r["LBTESTCD"] == "HBA1CHGB" and r["LBSTRESN"]
    ]

    by_subject: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in hba1c:
        by_subject[row["USUBJID"]][row["VISIT"]] = row

    # Only use subjects that have all 9 visits (full data)
    full_visit_set = {
        "SCREENING 1", "WEEK 2", "WEEK 6", "WEEK 8",
        "WEEK 12", "WEEK 16", "WEEK 20", "WEEK 24", "WEEK 26",
    }
    subjects = sorted(
        s for s, v in by_subject.items()
        if full_visit_set <= set(v.keys())
    )  # stable order: 01-701-1015, 01-701-1028, 01-701-1034

    cases: list[dict] = []
    for case_num, (subj_idx, visit_subset) in enumerate(TASK3_CASES, start=1):
        subject = subjects[subj_idx]
        visits_map = by_subject[subject]

        # Only use visits that exist in the data
        ordered_visits = [v for v in visit_subset if v in visits_map]
        if len(ordered_visits) < 2:
            continue

        base_row = visits_map["SCREENING 1"]
        base = float(base_row["LBSTRESN"])

        input_data = [
            {
                "USUBJID": visits_map[v]["USUBJID"],
                "VISIT": v,
                "LBTESTCD": visits_map[v]["LBTESTCD"],
                "LBTEST": visits_map[v]["LBTEST"],
                "LBSTRESN": float(visits_map[v]["LBSTRESN"]),
                "LBSTRESU": visits_map[v]["LBSTRESU"],
                "LBBLFL": "" if visits_map[v].get("LBBLFL", "") in {"", "NA"} else visits_map[v]["LBBLFL"],
            }
            for v in ordered_visits
        ]

        ground_truth = []
        for v in ordered_visits:
            row = visits_map[v]
            aval = float(row["LBSTRESN"])
            chg = round(aval - base, 2)
            pchg = round(((aval - base) / base) * 100, 2) if base else 0.0
            is_baseline = v == "SCREENING 1"
            ground_truth.append(
                {
                    "USUBJID": row["USUBJID"],
                    "VISIT": v,
                    "PARAM": f"{row['LBTEST']} [{row['LBSTRESU']}]",
                    "AVAL": aval,
                    "BASE": base,
                    "CHG": chg,
                    "PCHG": pchg,
                    "ABLFL": "Y" if is_baseline else "",
                    "ANL01FL": "" if is_baseline else "Y",
                }
            )

        n_visits = len(ordered_visits)
        if n_visits <= 3:
            difficulty = "easy"
        elif n_visits <= 4:
            difficulty = "medium"
        elif n_visits <= 5:
            difficulty = "hard"
        else:
            difficulty = "very hard"

        cases.append(
            {
                "case_id": f"PV_T3_{case_num}",
                "task_description": (
                    f"Derive an ADLB-style analysis dataset for subject {subject} "
                    f"using the {n_visits} supplied HbA1c SDTM laboratory rows. "
                    "Return a JSON array with one record per input row. "
                    "Every record must contain all 9 required fields."
                ),
                "study_context": TASK3_STUDY_CONTEXT,
                "difficulty": difficulty,
                "input_data": input_data,
                "ground_truth": ground_truth,
            }
        )

    return {
        "task_name": "task3_sdtm_to_adam",
        "description": (
            "Pharmaverse HbA1c lab records converted to ADLB-style outputs. "
            "10 cases with 3-9 visits each (easy to very hard)."
        ),
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dm_rows = read_csv(SDTM_REPO / "inst" / "extdata" / "dm.csv")
    ae_rows = read_csv(SDTM_REPO / "inst" / "extdata" / "ae.csv")
    lb_rows = read_csv(SDTM_REPO / "inst" / "extdata" / "lb_metabolic.csv")

    task1 = build_task1_cases(dm_rows)
    task2 = build_task2_cases(ae_rows)
    task3 = build_task3_cases(lb_rows)

    datasets = {
        "task1_cases.json": task1,
        "task2_cases.json": task2,
        "task3_cases.json": task3,
    }

    for filename, payload in datasets.items():
        output_path = OUT_DIR / filename
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        n = len(payload["cases"])
        print(f"Wrote {output_path}  ({n} cases)")


if __name__ == "__main__":
    main()
