"""Data contract for a future Brazilian validation dataset (one row per student x discipline x term).

It is the request specification to send to an institution AFTER authorisation, and a validator
for the delivered file. Field names are neutral (English) and map to the institution's schema.
LGPD: minimum necessary fields, pseudonymised identifiers, sensitive data only for auditing.
"""

from __future__ import annotations

import pandas as pd

REQUIRED = {
    "student_pseudo_id": {"type": "string", "description": "Identificador pseudonimizado e estável (hash com segredo da instituição)."},
    "institution_id": {"type": "string", "description": "Código da instituição (ou campus)."},
    "course_id": {"type": "string", "description": "Curso/programa (pseudonimizado se necessário)."},
    "component_id": {"type": "string", "description": "Disciplina/componente curricular."},
    "class_id": {"type": "string", "description": "Turma."},
    "term": {"type": "string", "pattern": r"^\d{4}\.[12]$", "description": "Período letivo, ex.: 2019.1."},
    "semester_start_date": {"type": "date", "description": "Início das aulas do período (calendário)."},
    "enrollment_date": {"type": "date", "description": "Data da matrícula na turma."},
    "final_status": {"type": "category",
                     "values": ["APROVADO", "REPROVADO_NOTA", "REPROVADO_FALTA", "REPROVADO_NOTA_FALTA",
                                "CANCELADO", "TRANCADO", "OUTRO"],
                     "description": "Situação final no componente."},
    "final_status_date": {"type": "date", "description": "Data da alteração de situação (para posicionar cancelamentos no tempo)."},
}

OPTIONAL = {
    "ava_log_timestamp": {"type": "datetime", "description": "Acesso ao AVA (linhas de log em tabela separada)."},
    "ava_log_student": {"type": "string", "description": "student_pseudo_id do log."},
    "ava_log_resource": {"type": "string", "description": "Recurso acessado."},
    "ava_log_resource_type": {"type": "string", "description": "Tipo do recurso (taxonomia do AVA)."},
    "activity_due_date": {"type": "date", "description": "Prazo de cada atividade avaliativa."},
    "submission_timestamp": {"type": "datetime", "description": "Data/hora de entrega da atividade."},
    "activity_weight": {"type": "number", "description": "Peso da atividade."},
    "absences_by_date": {"type": "number", "description": "Faltas registradas com data (para cortes semanais)."},
    "unit_grade": {"type": "number", "description": "Nota por unidade COM data de lançamento."},
    "birth_year": {"type": "integer", "description": "Ano de nascimento ou, preferencialmente, faixa etária."},
    "prior_education": {"type": "category", "description": "Escolaridade prévia."},
    "previous_enrollments_same_component": {"type": "integer", "description": "Matrículas anteriores no componente."},
    "planned_assessments": {"type": "integer", "description": "Nº de avaliações previstas."},
    "semester_end_date": {"type": "date", "description": "Fim do período letivo."},
}

AUDIT_ONLY = {
    "gender": {"type": "category", "description": "Apenas auditoria de equidade (Q16)."},
    "disability": {"type": "category", "description": "Apenas auditoria; dado pessoal sensível (LGPD art. 5º, II e art. 11)."},
    "socioeconomic_indicator": {"type": "category", "description": "Apenas auditoria (Q17)."},
}

FORBIDDEN = {"name", "cpf", "email", "phone", "address", "matricula", "nome_discente", "rg", "birth_date"}


def validate(df: pd.DataFrame) -> dict:
    """Check a delivered file against the contract. Returns problems; never modifies data."""
    cols = set(df.columns)
    problems = []
    missing = sorted(set(REQUIRED) - cols)
    if missing:
        problems.append({"type": "missing_required", "columns": missing})
    pii = sorted(cols & FORBIDDEN)
    if pii:
        problems.append({"type": "forbidden_personal_data", "columns": pii})
    if "term" in cols:
        bad = ~df["term"].astype(str).str.match(REQUIRED["term"]["pattern"])
        if bad.any():
            problems.append({"type": "invalid_term", "rows": int(bad.sum())})
    if "final_status" in cols:
        bad = ~df["final_status"].isin(REQUIRED["final_status"]["values"])
        if bad.any():
            problems.append({"type": "invalid_final_status", "rows": int(bad.sum())})
    if {"final_status", "final_status_date"} <= cols:
        undated = (df["final_status"] == "CANCELADO") & df["final_status_date"].isna()
        if undated.any():
            problems.append({"type": "cancellation_without_date", "rows": int(undated.sum())})
    unknown = sorted(cols - set(REQUIRED) - set(OPTIONAL) - set(AUDIT_ONLY) - FORBIDDEN)
    return {"valid": not problems, "problems": problems, "unrecognised_columns": unknown,
            "optional_present": sorted(cols & set(OPTIONAL)), "audit_present": sorted(cols & set(AUDIT_ONLY))}
