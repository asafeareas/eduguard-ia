"""Data-leakage audit: variable catalog + empirical checks.

Classes:
    PERMITTED   — available before or at the prediction moment.
    CONDITIONAL — usable only under a temporal filter or documented assumption.
    PROHIBITED  — uses future information or directly reveals the outcome.

The catalogs are the single source of truth for docs/DATA_LEAKAGE_AUDIT.md.
Nothing is permanently dropped here: the catalog only documents the decision.
"""

from __future__ import annotations

import pandas as pd

PERMITTED, CONDITIONAL, PROHIBITED = "PERMITTED", "CONDITIONAL", "PROHIBITED"


def _v(table, variable, cls, justification, **extra):
    return {"table": table, "variable": variable, "class": cls, "justification": justification, **extra}


OULAD_CATALOG = [
    _v("studentInfo", "code_module / code_presentation", PERMITTED,
       "Known at registration. Use for grouping/normalisation, not as a shortcut for outcome."),
    _v("studentInfo", "gender", PERMITTED, "Declared at registration. Fairness review required before use."),
    _v("studentInfo", "region", PERMITTED, "Declared at registration."),
    _v("studentInfo", "highest_education", PERMITTED, "Prior education, known at registration."),
    _v("studentInfo", "imd_band", PERMITTED, "Area deprivation index of the address; 1,111 missing values."),
    _v("studentInfo", "age_band", PERMITTED, "Known at registration."),
    _v("studentInfo", "disability", PERMITTED, "Declared at registration. Sensitive: fairness review required."),
    _v("studentInfo", "num_of_prev_attempts", PERMITTED, "Counts PAST attempts at the module."),
    _v("studentInfo", "studied_credits", CONDITIONAL,
       "Total credits the student is taking; the snapshot date is not documented and the value may "
       "include modules added/dropped during the year."),
    _v("studentInfo", "final_result", PROHIBITED, "It IS the outcome (label source)."),
    _v("studentRegistration", "date_registration", PERMITTED,
       "Known once the student registers; only defined for students registered by the cutoff."),
    _v("studentRegistration", "date_registration IS MISSING", CONDITIONAL,
       "45 rows; 86.7% of them are Withdrawn. Missingness may be an artefact of the withdrawal "
       "process (record cleanup), i.e. outcome-revealing. Do not use a missing-indicator."),
    _v("studentRegistration", "date_unregistration", PROHIBITED,
       "Defines the withdrawal event. Allowed ONLY to filter who is still registered at the cutoff "
       "(population definition), never as a feature."),
    _v("courses", "module_presentation_length", PERMITTED, "Planned length, published in advance."),
    _v("assessments", "assessment_type / weight / date (due date)", PERMITTED,
       "Published assessment calendar. 11 exam dates are missing."),
    _v("vle", "activity_type / week_from / week_to", PERMITTED,
       "Course design metadata (82% of week_from/week_to missing)."),
    _v("studentVle", "sum_click, id_site, date (date <= cutoff)", CONDITIONAL,
       "Each interaction has its own day stamp: usable only when date <= cutoff_day."),
    _v("studentVle", "any aggregate over the whole presentation", PROHIBITED,
       "Totals, last activity day, engagement slope over the full presentation all embed the future "
       "(and the absence of clicks after withdrawal)."),
    _v("studentAssessment", "date_submitted (<= cutoff)", CONDITIONAL,
       "Usable only for submissions on or before the cutoff."),
    _v("studentAssessment", "score (submission <= cutoff)", CONDITIONAL,
       "OULAD has no score RELEASE date; assuming the score is known on submission day is optimistic. "
       "Phase 03 must test a release lag (e.g. +7/+14 days)."),
    _v("studentAssessment", "is_banked == 1", CONDITIONAL,
       "Result carried over from a previous presentation; date semantics undocumented. Excluded from "
       "as-of aggregates for now."),
    _v("studentAssessment", "Exam score", PROHIBITED,
       "Final exam happens at the end of the presentation: future at any early cutoff."),
    _v("derived", "count/mean of assessments over the whole presentation", PROHIBITED,
       "Missing submissions after withdrawal mechanically encode the outcome."),
    _v("derived", "presentation-level withdrawal rate of the SAME presentation", PROHIBITED,
       "Target encoding with the evaluated cohort's own labels."),
    _v("derived", "statistics from PREVIOUS presentations (e.g. module historic withdrawal rate)", CONDITIONAL,
       "Allowed if computed only from presentations that finished before the predicted one starts."),
    _v("derived", "per-module click normalisation (percentiles, z-scores)", CONDITIONAL,
       "Must use only data <= cutoff and be fitted on training presentations only."),
    _v("design", "same id_student across presentations", CONDITIONAL,
       "Some students appear in more than one presentation (count in research/inspection/"
       "phase02_oulad.json): train/test must split by student and/or by time, otherwise the model "
       "sees the same person on both sides."),
]


def _u(variable, s1, s2, justification):
    """UCI entry. s1/s2 = class for 1st-semester / 2nd-semester early cutoffs."""
    return {"variable": variable, "class_sem1_weeks_2_6": s1, "class_sem2_weeks_2_6": s2,
            "justification": justification}


_ENROLL = ("Marital status", "Application mode", "Application order", "Course",
           "Daytime/evening attendance", "Previous qualification", "Previous qualification (grade)",
           "Nacionality", "Mother's qualification", "Father's qualification", "Mother's occupation",
           "Father's occupation", "Admission grade", "Displaced", "Educational special needs",
           "Gender", "Age at enrollment", "International")

UCI_CATALOG = (
    [_u(v, PERMITTED, PERMITTED, "Documented by the authors as known at enrollment.") for v in _ENROLL]
    + [
        _u("Debtor", CONDITIONAL, CONDITIONAL,
           "Collection date not documented; could reflect debt accumulated after leaving."),
        _u("Tuition fees up to date", CONDITIONAL, CONDITIONAL,
           "Collection date not documented; 32% of Dropout vs 1% of Graduate are not up to date — "
           "strong separation compatible with post-dropout measurement. Highest-risk variable."),
        _u("Scholarship holder", CONDITIONAL, CONDITIONAL,
           "Collection date not documented; scholarships can be granted or lost during the course."),
        _u("Curricular units 1st sem (enrolled)", CONDITIONAL, PERMITTED,
           "Set at course registration for the semester; the extraction moment is undocumented."),
        _u("Curricular units 1st sem (credited)", CONDITIONAL, PERMITTED,
           "Credit transfers are usually decided at enrollment; timing undocumented."),
        _u("Curricular units 1st sem (evaluations)", PROHIBITED, PERMITTED,
           "Counted at the END of semester 1. Students with units enrolled and 0 evaluations are 100% "
           "Dropout: the variable records the abandonment itself."),
        _u("Curricular units 1st sem (approved)", PROHIBITED, PERMITTED, "End-of-semester-1 result."),
        _u("Curricular units 1st sem (grade)", PROHIBITED, PERMITTED, "End-of-semester-1 average."),
        _u("Curricular units 1st sem (without evaluations)", PROHIBITED, PERMITTED,
           "End-of-semester-1 result."),
        _u("Curricular units 2nd sem (enrolled)", PROHIBITED, CONDITIONAL,
           "Known at the start of semester 2, but value 0 can mean the student already left "
           "(outcome-revealing)."),
        _u("Curricular units 2nd sem (credited)", PROHIBITED, CONDITIONAL, "Same as above."),
        _u("Curricular units 2nd sem (evaluations)", PROHIBITED, PROHIBITED, "End-of-semester-2 result."),
        _u("Curricular units 2nd sem (approved)", PROHIBITED, PROHIBITED, "End-of-semester-2 result."),
        _u("Curricular units 2nd sem (grade)", PROHIBITED, PROHIBITED, "End-of-semester-2 average."),
        _u("Curricular units 2nd sem (without evaluations)", PROHIBITED, PROHIBITED,
           "End-of-semester-2 result."),
        _u("Unemployment rate", CONDITIONAL, CONDITIONAL,
           "Reference year undocumented. Only 10 distinct values: acts as a cohort identifier."),
        _u("Inflation rate", CONDITIONAL, CONDITIONAL, "Same as Unemployment rate."),
        _u("GDP", CONDITIONAL, CONDITIONAL, "Same as Unemployment rate."),
        _u("Target", PROHIBITED, PROHIBITED, "Outcome label."),
    ]
)


def catalog_frame(catalog: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(catalog)


def uci_outcome_revealing_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Share of Dropout under conditions that look like 'the abandonment already happened'."""
    s1, s2 = "Curricular units 1st sem ", "Curricular units 2nd sem "
    rules = {
        "1st sem: enrolled>0 and evaluations==0": (df[s1 + "(enrolled)"] > 0) & (df[s1 + "(evaluations)"] == 0),
        "1st sem: approved==0": df[s1 + "(approved)"] == 0,
        "2nd sem: enrolled==0": df[s2 + "(enrolled)"] == 0,
        "2nd sem: enrolled>0 and evaluations==0": (df[s2 + "(enrolled)"] > 0) & (df[s2 + "(evaluations)"] == 0),
        "2nd sem: approved==0": df[s2 + "(approved)"] == 0,
        "Tuition fees up to date == 0": df["Tuition fees up to date"] == 0,
        "Debtor == 1": df["Debtor"] == 1,
    }
    rows = []
    for name, mask in rules.items():
        sub = df.loc[mask, "Target"]
        rows.append({
            "rule": name,
            "n": int(mask.sum()),
            "share_dropout": round(float((sub == "Dropout").mean()), 4) if len(sub) else float("nan"),
            "share_enrolled": round(float((sub == "Enrolled").mean()), 4) if len(sub) else float("nan"),
            "share_graduate": round(float((sub == "Graduate").mean()), 4) if len(sub) else float("nan"),
        })
    return pd.DataFrame(rows)
