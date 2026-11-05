"""Feature catalog: the single source of truth for every modelling column.

Each feature declares where it comes from, from which moment it is known,
which cutoffs it may be used at, its type (for later encoding fitted on the
training split only) and its expected domain (for quality checks).

Status:
    "included"    — part of the main dataset for the listed cutoffs.
    "quarantined" — NOT produced in the main datasets (temporal availability
                    unproven or outcome-revealing). Listed for traceability.

Column roles in every processed dataset:
    ID_COLUMNS     — never predictive; kept for joins/validation/grouping.
    TARGET_COLUMNS — labels only.
    AUDIT_COLUMNS  — kept for verification; NEVER predictive (may contain the outcome).
"""

from __future__ import annotations

from dataclasses import dataclass, field

WEEKS = (2, 4, 6)


@dataclass(frozen=True)
class Feature:
    name: str
    group: str            # engagement | assessment | context | admission
    description: str
    formula: str
    source: str
    available_from: str   # human-readable moment
    cutoffs: tuple[int, ...]
    ftype: str            # numeric | count | binary | ordinal | nominal
    justification: str
    status: str = "included"
    sensitive: bool = False
    domain: dict = field(default_factory=dict)  # {"min":..,"max":..} or {"values": [...]}


def _f(*args, **kw) -> Feature:
    return Feature(*args, **kw)


ALL = WEEKS
S46 = (4, 6)
NONNEG = {"min": 0}
RATIO = {"min": 0, "max": 1}

OULAD_FEATURES: list[Feature] = [
    # ---------------------------------------------------------------- engagement
    _f("eng_total_clicks", "engagement", "Total de cliques no AVA até o corte (inclui antes do início).",
       "Σ sum_click, date ≤ corte", "studentVle", "cada clique, no seu dia", ALL, "count",
       "Volume de engajamento; separação 0,24–0,27 na Fase 02.", domain=NONNEG),
    _f("eng_any_activity", "engagement", "1 se houve ao menos um clique até o corte.",
       "1[eng_total_clicks > 0]", "studentVle", "cada clique", ALL, "binary",
       "Distingue 'sem atividade' de 'pouca atividade'; 8% dos Withdrawn sem cliques em S2.",
       domain={"values": [0, 1]}),
    _f("eng_active_days", "engagement", "Dias distintos com atividade até o corte.",
       "nº de dias distintos com clique, date ≤ corte", "studentVle", "cada clique", ALL, "count",
       "Regularidade; proxy de frequência no EAD (não é presença).", domain=NONNEG),
    _f("eng_clicks_per_active_day", "engagement", "Média de cliques por dia ativo.",
       "eng_total_clicks / eng_active_days (0 se nenhum dia ativo)", "studentVle", "cada clique", ALL,
       "numeric", "Intensidade por sessão, independente da regularidade.", domain=NONNEG),
    _f("eng_clicks_per_elapsed_day", "engagement", "Cliques por dia decorrido da apresentação.",
       "Σ sum_click (0 ≤ date ≤ corte) / (corte + 1)", "studentVle", "cada clique", ALL, "numeric",
       "Normaliza pelo tempo decorrido, comparável entre cortes.", domain=NONNEG),
    _f("eng_clicks_pre_start", "engagement", "Cliques antes do dia 0.",
       "Σ sum_click, date < 0", "studentVle", "antes do início", ALL, "count",
       "Preparação prévia; separação 0,21 em S2.", domain=NONNEG),
    _f("eng_active_days_pre_start", "engagement", "Dias ativos antes do dia 0.",
       "nº de dias distintos com clique, date < 0", "studentVle", "antes do início", ALL, "count",
       "Complementa o volume pré-início com regularidade.", domain=NONNEG),
    _f("eng_active_weeks", "engagement", "Semanas (1..k) com ao menos um clique.",
       "Σ_{w=1..k} 1[cliques_w > 0]", "studentVle", "fim de cada semana", ALL, "count",
       "Persistência semanal.", domain=NONNEG),
    _f("eng_inactive_weeks", "engagement", "Semanas (1..k) sem nenhum clique.",
       "k − eng_active_weeks", "studentVle", "fim de cada semana", ALL, "count",
       "Inatividade semanal cresce de 41% para 65% antes do cancelamento (Fase 02).", domain=NONNEG),
    _f("eng_active_weeks_ratio", "engagement", "Percentual de semanas com atividade.",
       "eng_active_weeks / k", "studentVle", "fim de cada semana", ALL, "numeric",
       "Comparável entre cortes com k diferente.", domain=RATIO),
    _f("eng_days_since_last_activity", "engagement", "Dias desde o último clique (recência).",
       "corte − último dia com clique ≤ corte; se nunca clicou: corte + 1", "studentVle", "cada clique",
       ALL, "numeric", "Recência: separação 0,21–0,27. Valor para 'nunca clicou' é determinístico "
       "e acompanhado de eng_any_activity.", domain={"min": 0}),
    _f("eng_first_activity_day", "engagement", "Dia do primeiro clique (negativo = antes do início).",
       "mín(date) ≤ corte; se nunca clicou: corte + 1", "studentVle", "cada clique", ALL, "numeric",
       "Início tardio do engajamento como sinal de risco.", domain={"min": -400}),
    _f("eng_distinct_sites", "engagement", "Materiais distintos acessados.",
       "nº de id_site distintos, date ≤ corte", "studentVle", "cada clique", ALL, "count",
       "Amplitude de exploração do curso.", domain=NONNEG),
    _f("eng_distinct_activity_types", "engagement", "Tipos de material distintos acessados.",
       "nº de activity_type distintos, date ≤ corte", "studentVle + vle", "cada clique", ALL, "count",
       "Diversidade de atividades.", domain={"min": 0, "max": 20}),
    _f("eng_activity_type_entropy", "engagement", "Entropia de Shannon dos cliques por tipo de material.",
       "−Σ p_t log2 p_t, p_t = cliques do tipo t / total (0 se sem cliques)", "studentVle + vle",
       "cada clique", ALL, "numeric", "Diversidade ponderada pelo volume.", domain={"min": 0, "max": 4.33}),
    _f("eng_clicks_last_14_days", "engagement", "Cliques nos 14 dias até o corte.",
       "Σ sum_click, corte − 13 ≤ date ≤ corte", "studentVle", "cada clique", S46, "count",
       "Engajamento recente. Em S2 seria idêntico ao total do período → só S4/S6.", domain=NONNEG),
    _f("eng_active_days_last_14_days", "engagement", "Dias ativos nos 14 dias até o corte.",
       "nº de dias com clique, corte − 13 ≤ date ≤ corte", "studentVle", "cada clique", S46, "count",
       "Frequência de acesso nas últimas duas semanas.", domain={"min": 0, "max": 14}),
    _f("eng_weekly_clicks_std", "engagement", "Desvio-padrão dos cliques semanais (semanas 1..k).",
       "desvio-padrão populacional de cliques_w, w = 1..k (zeros incluídos)", "studentVle",
       "fim de cada semana", ALL, "numeric", "Variação entre semanas.", domain=NONNEG),
    _f("eng_weekly_clicks_cv", "engagement", "Coeficiente de variação dos cliques semanais.",
       "eng_weekly_clicks_std / média(cliques_w) (0 se média 0)", "studentVle", "fim de cada semana",
       ALL, "numeric", "Irregularidade relativa ao próprio volume.", domain=NONNEG),
    _f("eng_weekly_trend_slope", "engagement", "Inclinação (MQO) dos cliques semanais ao longo de 1..k.",
       "cov(w, cliques_w) / var(w), w = 1..k", "studentVle", "fim de cada semana", ALL, "numeric",
       "Tendência de crescimento/redução: desengajamento progressivo."),
    _f("eng_last_week_change", "engagement", "Variação de cliques da semana k−1 para a semana k.",
       "cliques_k − cliques_{k−1}", "studentVle", "fim da semana k", S46, "numeric",
       "Queda abrupta recente. Em S2 (k = 2) é algebricamente idêntica a eng_weekly_trend_slope "
       "→ só S4/S6."),
    _f("eng_clicks_pct_in_presentation", "engagement",
       "Percentil dos cliques acumulados entre os estudantes elegíveis da mesma apresentação no corte.",
       "rank percentual de eng_total_clicks dentro de (módulo, apresentação, corte)", "studentVle",
       "no corte (usa só dados ≤ corte de colegas elegíveis; nenhum rótulo)", ALL, "numeric",
       "Engajamento relativo ao contexto do módulo (CCC ≠ GGG).", domain={"min": 0, "max": 1}),
    *[
        _f(f"eng_clicks_week_{w:02d}", "engagement", f"Cliques na semana {w} (dias {7 * (w - 1)}–{7 * w - 1}).",
           f"Σ sum_click, {7 * (w - 1)} ≤ date ≤ {7 * w - 1}", "studentVle", f"fim da semana {w}",
           tuple(c for c in WEEKS if c >= w), "count", "Trajetória semanal completa até o corte.",
           domain=NONNEG)
        for w in range(1, 7)
    ],
    # ---------------------------------------------------------------- assessment
    _f("asm_n_due", "assessment", "Avaliações não-finais com prazo ≤ corte (calendário).",
       "nº de assessments (TMA/CMA) com date ≤ corte", "assessments", "calendário publicado", S46, "count",
       "OULAD não tem data de abertura: 'disponibilizada' é aproximada por 'prazo vencido'. "
       "Em S2 só 4,6% das apresentações têm prazo vencido → excluída de S2.", domain=NONNEG),
    _f("asm_n_submitted", "assessment", "Avaliações não-finais entregues até o corte (não-banked).",
       "nº de entregas com date_submitted ≤ corte, is_banked = 0", "studentAssessment",
       "dia da entrega", S46, "count", "Participação; inclui entregas antecipadas.", domain=NONNEG),
    _f("asm_n_due_submitted", "assessment", "Avaliações com prazo ≤ corte e entregues até o corte.",
       "nº de (prazo ≤ corte) ∧ (date_submitted ≤ corte)", "studentAssessment", "dia da entrega",
       S46, "count", "Base da taxa de entrega.", domain=NONNEG),
    _f("asm_submission_rate", "assessment", "Percentual de avaliações vencidas que foram entregues.",
       "asm_n_due_submitted / asm_n_due (vazio se asm_n_due = 0)", "studentAssessment",
       "dia da entrega", S46, "numeric", "Indefinida sem prazo vencido: deixada ausente para "
       "imputação ajustada só no treino.", domain=RATIO),
    _f("asm_n_missing", "assessment", "Avaliações vencidas não entregues até o corte.",
       "asm_n_due − asm_n_due_submitted", "studentAssessment", "prazo", S46, "count",
       "22,6% dos Withdrawn não entregam a 1ª avaliação (Pass: 1%).", domain=NONNEG),
    _f("asm_missed_weight", "assessment", "Soma dos pesos (%) das avaliações vencidas não entregues.",
       "Σ weight das avaliações em asm_n_missing", "assessments + studentAssessment", "prazo", S46,
       "numeric", "Quanto da nota já foi perdido por não entrega.", domain={"min": 0, "max": 100}),
    _f("asm_n_on_time", "assessment", "Entregas até o corte feitas dentro do prazo.",
       "nº de entregas com date_submitted ≤ min(prazo, corte)", "studentAssessment", "dia da entrega",
       S46, "count", "Pontualidade.", domain=NONNEG),
    _f("asm_n_late", "assessment", "Entregas até o corte feitas após o prazo.",
       "nº de entregas com prazo < date_submitted ≤ corte", "studentAssessment", "dia da entrega",
       S46, "count", "Atraso como sinal de dificuldade.", domain=NONNEG),
    _f("asm_mean_days_before_deadline", "assessment",
       "Média de dias entre a entrega e o prazo (positivo = antecipada).",
       "média(prazo − date_submitted) das entregas ≤ corte (vazio se nenhuma)", "studentAssessment",
       "dia da entrega", S46, "numeric",
       "Substitui 'dias entre disponibilização e submissão': a data de abertura não existe na OULAD."),
    _f("asm_first_due_passed", "assessment", "1 se o prazo da 1ª avaliação não-final já venceu no corte.",
       "1[prazo da 1ª avaliação ≤ corte]", "assessments", "calendário", S46, "binary",
       "Contexto para interpretar asm_first_submitted.", domain={"values": [0, 1]}),
    _f("asm_first_submitted", "assessment", "1 se a 1ª avaliação não-final foi entregue até o corte.",
       "1[date_submitted(1ª avaliação) ≤ corte]", "studentAssessment", "dia da entrega", S46, "binary",
       "Participação nas primeiras avaliações.", domain={"values": [0, 1]}),
    # ---------------------------------------------------------------- context
    _f("ctx_code_module", "context", "Módulo.", "code_module", "courses", "antes do início", ALL,
       "nominal", "Taxa de Withdrawn de 11,5% a 44,5% entre módulos. Risco: aprender a turma.",
       domain={"values": list("AAA BBB CCC DDD EEE FFF GGG".split())}),
    _f("ctx_presentation_start", "context", "Mês de início da apresentação (B = fevereiro, J = outubro).",
       "último caractere de code_presentation", "courses", "antes do início", ALL, "nominal",
       "Sazonalidade sem expor o ano (o ano será usado para validação temporal).",
       domain={"values": ["B", "J"]}),
    _f("ctx_module_length", "context", "Duração planejada da apresentação (dias).",
       "module_presentation_length", "courses", "antes do início", ALL, "numeric",
       "Planejada e publicada.", domain={"min": 200, "max": 300}),
    _f("ctx_n_planned_assessments", "context", "Nº de avaliações não-finais planejadas.",
       "nº de TMA/CMA da apresentação", "assessments", "antes do início", ALL, "count",
       "Carga avaliativa planejada.", domain=NONNEG),
    # ---------------------------------------------------------------- admission
    _f("adm_gender", "admission", "Gênero declarado.", "gender", "studentInfo", "matrícula", ALL,
       "nominal", "Associação fraca (V = 0,02). Sensível: avaliar equidade.", sensitive=True,
       domain={"values": ["M", "F"]}),
    _f("adm_region", "admission", "Região de residência.", "region", "studentInfo", "matrícula", ALL,
       "nominal", "V = 0,07."),
    _f("adm_highest_education", "admission", "Escolaridade prévia (ordinal 0–4).",
       "No Formal quals=0 … Post Graduate Qualification=4", "studentInfo", "matrícula", ALL, "ordinal",
       "42,9% × 23,6% de Withdrawn entre extremos.", domain={"min": 0, "max": 4}),
    _f("adm_imd_band", "admission", "Faixa do índice de privação (ordinal 0–9; 0 = mais privado).",
       "0-10%=0 … 90-100%=9; ausente mantido ausente", "studentInfo", "matrícula", ALL, "ordinal",
       "37,2% × 25,9%. 1.111 ausentes: imputação ajustada só no treino.", domain={"min": 0, "max": 9}),
    _f("adm_age_band", "admission", "Faixa etária (ordinal 0–2).", "0-35=0, 35-55=1, 55<=2",
       "studentInfo", "matrícula", ALL, "ordinal", "V = 0,06.", domain={"min": 0, "max": 2}),
    _f("adm_disability", "admission", "Declarou deficiência.", "1[disability = Y]", "studentInfo",
       "matrícula", ALL, "binary", "39,4% × 30,3%. Sensível: avaliar equidade.", sensitive=True,
       domain={"values": [0, 1]}),
    _f("adm_num_prev_attempts", "admission", "Tentativas anteriores no módulo.", "num_of_prev_attempts",
       "studentInfo", "matrícula", ALL, "count", "Histórico anterior ao início.", domain={"min": 0, "max": 10}),
    _f("adm_registration_lead_days", "admission", "Antecedência da matrícula (dias antes do início).",
       "−date_registration (negativo = matrícula após o início); ausente mantido ausente",
       "studentRegistration", "matrícula", ALL, "numeric",
       "≤ −90 dias: 38,2% × após o início: 20,9%. Sem indicador de ausência (ver quarentena)."),
    # ---------------------------------------------------------------- quarantined
    _f("studied_credits", "admission", "Créditos cursados no ano.", "studied_credits", "studentInfo",
       "desconhecido", (), "count", "Data de registro não documentada (pode mudar no ano). "
       "Forte associação (51,3% × 25,4%) — quarentena até comprovação.", status="quarantined"),
    _f("asm_mean_score", "assessment", "Nota média das avaliações.", "média(score)", "studentAssessment",
       "desconhecido (sem data de divulgação)", (), "numeric",
       "Data de divulgação inexistente: não pode ser assumida conhecida na entrega.", status="quarantined"),
    _f("asm_banked_results", "assessment", "Resultados aproveitados de apresentação anterior.",
       "is_banked = 1", "studentAssessment", "desconhecido", (), "count",
       "Semântica de data não documentada.", status="quarantined"),
    _f("adm_registration_missing", "admission", "Indicador de data de matrícula ausente.",
       "1[date_registration ausente]", "studentRegistration", "—", (), "binary",
       "86,7% dos 45 casos são Withdrawn: ausência possivelmente causada pelo cancelamento.",
       status="quarantined"),
    _f("ctx_presentation_year", "context", "Ano da apresentação.", "code_presentation[:4]", "courses",
       "antes do início", (), "numeric",
       "Mantido como coluna de auditoria para validação temporal; como feature impediria "
       "generalizar para anos futuros.", status="quarantined"),
]

UCI_NOMINAL = ("Marital status", "Application mode", "Course", "Previous qualification", "Nacionality",
               "Mother's qualification", "Father's qualification", "Mother's occupation",
               "Father's occupation")
UCI_BINARY = ("Daytime/evening attendance", "Displaced", "Educational special needs", "Gender",
              "International")
UCI_NUMERIC = ("Application order", "Previous qualification (grade)", "Admission grade", "Age at enrollment")

UCI_ADMISSION_FEATURES: list[str] = [*UCI_NOMINAL, *UCI_BINARY, *UCI_NUMERIC]

UCI_QUARANTINED: dict[str, str] = {
    "Scholarship holder": "Data de coleta não documentada; bolsas podem ser concedidas/perdidas no curso.",
    "Debtor": "Data de coleta não documentada; pode refletir dívida pós-abandono.",
    "Tuition fees up to date": "Data de coleta não documentada; 86,6% Dropout entre os 'não em dia'.",
    "Curricular units 1st sem (enrolled)": "Momento da extração não documentado.",
    "Curricular units 1st sem (credited)": "Momento da extração não documentado.",
    "Unemployment rate": "Ano de referência não documentado; mantido como auditoria (proxy de coorte).",
    "Inflation rate": "Idem.",
    "GDP": "Idem.",
}
UCI_PROHIBITED = [
    f"Curricular units {s} sem ({v})" for s in ("1st", "2nd")
    for v in ("evaluations", "approved", "grade", "without evaluations")
] + ["Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (credited)", "Target"]

UCI_DOMAINS = {
    "Admission grade": {"min": 0, "max": 200},
    "Previous qualification (grade)": {"min": 0, "max": 200},
    "Application order": {"min": 0, "max": 9},
    "Age at enrollment": {"min": 15, "max": 80},
    **{b: {"values": [0, 1]} for b in UCI_BINARY},
}

# Column roles -------------------------------------------------------------------------
OULAD_ID_COLUMNS = ["student_id", "presentation_id", "code_module", "code_presentation"]
OULAD_TARGET_COLUMNS = ["target_withdrawn", "target_adverse_outcome"]
OULAD_AUDIT_COLUMNS = ["cutoff_week", "cutoff_day", "presentation_year", "final_result",
                       "date_registration", "date_unregistration", "max_vle_day_used",
                       "max_submission_day_used"]
UCI_ID_COLUMNS = ["row_id"]
UCI_TARGET_COLUMNS = ["target_dropout"]
UCI_AUDIT_COLUMNS = ["original_target", "cohort_proxy"]


def oulad_features_for_week(week: int) -> list[Feature]:
    return [f for f in OULAD_FEATURES if f.status == "included" and week in f.cutoffs]


def feature_names_for_week(week: int) -> list[str]:
    return [f.name for f in oulad_features_for_week(week)]


def by_name() -> dict[str, Feature]:
    return {f.name: f for f in OULAD_FEATURES}
