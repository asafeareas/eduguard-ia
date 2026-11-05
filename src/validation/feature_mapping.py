"""OULAD -> Brazilian correspondence matrix (Q20 preparation). No Brazilian data is read here.

Status of each OULAD feature in a Brazilian setting:
    direct        — same concept, same definition, computable from expected academic records.
    transform     — related concept, but needs redefinition/re-scaling; the trained model's
                    coefficients would NOT keep their meaning -> not a valid input for direct transfer.
    no_equivalent — no counterpart expected in academic records (would require data that may not exist).
    excluded      — must not be used (methodological, model-specific or privacy reasons).

`requires` lists the Brazilian fields (see data_contract.REQUIRED/OPTIONAL) needed to compute it.
Equivalences are deliberately conservative: nothing is called `direct` unless the definition
carries over unchanged. Facts about UFRN open data come from Phase 01 (column headers read on
2026-11-05); nothing else about UFRN is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DIRECT, TRANSFORM, NO_EQUIVALENT, EXCLUDED = "direct", "transform", "no_equivalent", "excluded"


@dataclass(frozen=True)
class Mapping:
    feature: str
    status: str
    brazilian_equivalent: str
    requires: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


_ENG_NOTE = ("Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos "
             "dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente "
             "entre plataformas → redefinição e recalibração obrigatórias.")
_ASM_NOTE = ("Exige calendário de atividades (prazos) e data/hora de entrega. Os dados abertos da UFRN têm "
             "nota e faltas por unidade, sem datas (Fase 01). Só seria computável a partir de atividades "
             "entregues no AVA.")

MAPPINGS: list[Mapping] = [
    # engagement --------------------------------------------------------------------------
    *[Mapping(f, TRANSFORM, "Acessos ao AVA da disciplina até o corte", ("ava_log_timestamp", "ava_log_student"),
              _ENG_NOTE) for f in (
        "eng_total_clicks", "eng_any_activity", "eng_active_days", "eng_clicks_per_active_day",
        "eng_clicks_per_elapsed_day", "eng_active_weeks", "eng_inactive_weeks", "eng_active_weeks_ratio",
        "eng_days_since_last_activity", "eng_first_activity_day", "eng_clicks_last_14_days",
        "eng_active_days_last_14_days", "eng_weekly_clicks_std", "eng_weekly_clicks_cv",
        "eng_weekly_trend_slope", "eng_last_week_change", "eng_clicks_pct_in_presentation",
        *[f"eng_clicks_week_{w:02d}" for w in range(1, 7)])],
    Mapping("eng_clicks_pre_start", TRANSFORM, "Acessos ao AVA antes do início das aulas",
            ("ava_log_timestamp", "semester_start_date"), _ENG_NOTE + " No Brasil o AVA costuma abrir junto com as aulas."),
    Mapping("eng_active_days_pre_start", TRANSFORM, "Dias com acesso antes do início das aulas",
            ("ava_log_timestamp", "semester_start_date"), _ENG_NOTE),
    Mapping("eng_distinct_sites", TRANSFORM, "Recursos distintos acessados no AVA", ("ava_log_resource",), _ENG_NOTE),
    Mapping("eng_distinct_activity_types", TRANSFORM, "Tipos de recurso distintos (taxonomia do AVA local)",
            ("ava_log_resource_type",), "Os 20 tipos da OU (oucontent, forumng…) não correspondem aos módulos do "
            "Moodle/SIGAA; exige tabela de correspondência validada."),
    Mapping("eng_activity_type_entropy", TRANSFORM, "Entropia por tipo de recurso", ("ava_log_resource_type",),
            "Depende da taxonomia; não comparável sem harmonização."),
    # assessment --------------------------------------------------------------------------
    *[Mapping(f, TRANSFORM, "Atividades avaliativas com prazo e entrega datada", ("activity_due_date", "submission_timestamp"),
              _ASM_NOTE) for f in (
        "asm_n_due", "asm_n_submitted", "asm_n_due_submitted", "asm_submission_rate", "asm_n_missing",
        "asm_n_on_time", "asm_n_late", "asm_mean_days_before_deadline", "asm_first_due_passed",
        "asm_first_submitted")],
    Mapping("asm_missed_weight", TRANSFORM, "Peso das atividades vencidas não entregues",
            ("activity_due_date", "submission_timestamp", "activity_weight"),
            _ASM_NOTE + " Pesos raramente são registrados no sistema acadêmico."),
    # context -----------------------------------------------------------------------------
    Mapping("ctx_code_module", EXCLUDED, "Nenhuma (identidade dos 7 módulos da OU)", (),
            "Categorias AAA–GGG não existem fora da OU. O one-hot ignoraria valores novos, transformando "
            "silenciosamente todos os cursos brasileiros na categoria de referência. Não mapear."),
    Mapping("ctx_presentation_start", EXCLUDED, "Nenhuma (fevereiro/outubro do calendário britânico)", (),
            "Semestres brasileiros (1º/2º) não equivalem às apresentações B/J."),
    Mapping("ctx_module_length", TRANSFORM, "Duração do período letivo (dias)", ("semester_start_date", "semester_end_date"),
            "Módulos da OU têm de 234 a 269 dias; semestres brasileiros, cerca de 100 a 120 → fora da faixa de treino."),
    Mapping("ctx_n_planned_assessments", TRANSFORM, "Nº de avaliações/unidades previstas no plano da disciplina",
            ("planned_assessments",), "A UFRN registra 'unidade' (1–3) nas matrículas em componentes; não é o mesmo que TMA/CMA."),
    # admission ---------------------------------------------------------------------------
    Mapping("adm_region", EXCLUDED, "Nenhuma (13 regiões do Reino Unido)", (), "Sem correspondência geográfica."),
    Mapping("adm_highest_education", TRANSFORM, "Escolaridade prévia (ensino médio, superior incompleto…)",
            ("prior_education",), "As qualificações britânicas (A Level, HE…) exigem tabela de equivalência validada; "
            "a ordem 0–4 não é garantida."),
    Mapping("adm_age_band", DIRECT, "Faixa etária no início do período (0–35, 35–55, 55+)", ("birth_year",),
            "Mesma definição. Minimizar: enviar só a faixa, nunca a data de nascimento."),
    Mapping("adm_num_prev_attempts", TRANSFORM, "Matrículas anteriores na mesma disciplina",
            ("previous_enrollments_same_component",), "Definição de 'tentativa' varia (trancamento conta?)."),
    Mapping("adm_registration_lead_days", TRANSFORM, "Antecedência da matrícula na disciplina",
            ("enrollment_date", "semester_start_date"),
            "No Brasil a matrícula segue o calendário institucional (janelas fixas): a variação tem outro significado."),
    # audit-only (Q16/Q17) ------------------------------------------------------------------
    Mapping("adm_gender", EXCLUDED, "Sexo/gênero (apenas auditoria)", ("gender",), "Q16: nunca preditor."),
    Mapping("adm_disability", EXCLUDED, "Deficiência (apenas auditoria; dado sensível — LGPD art. 5º, II)",
            ("disability",), "Q16/Q18: nunca preditor; exige base legal específica (art. 11)."),
    Mapping("adm_imd_band", EXCLUDED, "Indicador socioeconômico (ex.: renda per capita, cota) — apenas auditoria",
            ("socioeconomic_indicator",), "Q17: removido do modelo principal; o IMD é britânico e sem equivalente direto."),
]

# Brazilian information with NO OULAD counterpart (relevant for re-training, not for transfer).
BRAZILIAN_ONLY = {
    "frequencia_faltas": "Faltas por unidade/total (UFRN: faltas_unidade, numero_total_faltas). Frequência é obrigatória no presencial (LDB, art. 47 §3º); o mínimo "
                         "(usualmente 75%) vem do regulamento de cada instituição — verificar.",
    "notas_por_unidade": "Notas parciais por unidade (UFRN: nota, unidade) — sem data de lançamento nos dados abertos.",
    "situacao_componente": "Aprovado / reprovado por nota / por falta / cancelado / trancado (UFRN: descricao).",
    "historico_semestres_anteriores": "Desempenho em períodos anteriores do mesmo curso.",
    "forma_de_ingresso": "SiSU, transferência, cotas (sensível: tratar como auditoria).",
}

# Phase 01 verified column headers of UFRN open data (public CKAN); used only to document coverage.
UFRN_OPEN_DATA_FIELDS = {
    "matriculas-componentes": ["id_turma", "discente", "id_curso", "unidade", "nota", "reposicao",
                               "faltas_unidade", "media_final", "numero_total_faltas", "descricao"],
    "situacoes-de-discentes": ["id_discente", "ano_ingresso", "periodo_ingresso", "nivel_ensino", "id_curso",
                               "curso", "situacao", "data_alteracao_situacao", "ano_alteracao_situacao",
                               "periodo_alteracao_situacao"],
}


def by_feature() -> dict[str, Mapping]:
    return {m.feature: m for m in MAPPINGS}
