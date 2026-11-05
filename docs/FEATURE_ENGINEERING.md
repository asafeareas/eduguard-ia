# Engenharia de Features — EduGuard-IA (Fase 03)

> 2026-11-05 · Branch `feat/feature-engineering`.
> Fonte única: `src/feature_engineering/catalog.py`. As tabelas abaixo foram **geradas a
> partir do catálogo** e um teste (`tests/test_phase03_docs.py`) garante que toda feature
> do catálogo esteja documentada aqui.
> Construção: `python -m src.feature_engineering.build_datasets`.
>
> **Nenhum modelo foi treinado.** Nenhuma imputação, codificação, normalização ou
> balanceamento foi aplicado: os valores ausentes ficam ausentes e as transformações
> serão ajustadas **somente no conjunto de treino** na próxima fase.

## 1. Regras gerais

| Regra | Implementação |
|---|---|
| Convenção temporal | Semana *k* termina no dia `7k − 1`: S2 = 13, S4 = 27, S6 = 41 (`temporal.cutoff_day_for_week`) |
| Eventos | Só cliques e entregas com data ≤ corte. Cliques antes do dia 0 contam |
| População | Matriculado até o corte e não cancelado até o corte (`temporal.population_at_cutoff`) |
| Rótulos inconsistentes | Excluídos de todos os cortes: 93 `Withdrawn` sem data de cancelamento e 9 não-`Withdrawn` com data (não podem ser posicionados no tempo) |
| Notas (`score`) | **Não usadas** (quarentena): a OULAD não registra quando a nota foi divulgada |
| Resultados *banked* | As 1.909 linhas têm `date_submitted = −1` (aproveitadas antes do início). Não entram como entrega nem como nota; a avaliação correspondente sai do denominador de "vencidas" daquele estudante (não conta como falta) |
| "Avaliação disponibilizada" | A OULAD não tem data de abertura. Aproximação: **prazo ≤ corte** (`asm_n_due`) |
| Valores para "sem atividade" | Determinísticos e documentados (ex.: recência = corte + 1), acompanhados de `eng_any_activity` |
| Tipos | Fixados pelo catálogo (`canonical_dtypes`): count/binary → int64, numeric/ordinal → float64, nominal → texto |
| Colunas não preditivas | IDs (`student_id`, `presentation_id`, módulo, apresentação), targets e auditoria. `split_roles()` devolve **apenas** as features |

## 2. Composição por corte

| Grupo | S2 (dia 13) | S4 (dia 27) | S6 (dia 41) | Observação |
|---|---|---|---|---|
| Engajamento digital | 21 | 26 | 28 | Semanais `eng_clicks_week_01..k`; janelas de 14 dias e variação semanal só a partir de S4 |
| Atividades e avaliações | 0 | 11 | 11 | Em S2 só 4,6% das apresentações têm prazo vencido → grupo excluído |
| Contexto do módulo | 4 | 4 | 4 | Conhecido antes do início |
| Ingresso e histórico | 8 | 8 | 8 | Conhecido na matrícula |
| **Total** | **33** | **49** | **51** | |

S2 usa engajamento + informações estáticas conhecidas desde a matrícula (dia ≤ 0), que não
são "posteriores" a nenhum corte. Não há nenhuma informação de avaliação em S2.

S6 **não** é S4 com mais registros: cada feature foi verificada individualmente
(`cutoffs` no catálogo) e todas são recalculadas do zero com o limite do dia 41.

## 3. Identidades e redundâncias conhecidas (mantidas de propósito)

A Fase 03 **não remove** variáveis por correlação; as redundâncias ficam registradas para a
seleção na fase de modelagem:

- `eng_active_weeks`, `eng_inactive_weeks` e `eng_active_weeks_ratio` são funções exatas umas
  das outras dentro de um corte (k é fixo). Mantidas porque o percentual é comparável entre cortes.
- `asm_submission_rate` × `asm_n_missing` (ρ ≈ −1,0), `asm_n_missing` × `asm_missed_weight` (0,98).
- `eng_total_clicks` × `eng_clicks_per_elapsed_day` (0,94–0,98).
- `eng_last_week_change` foi **retirada de S2**: com k = 2 ela é algebricamente idêntica a
  `eng_weekly_trend_slope`.

Lista completa por corte: `research/inspection/phase03_quality_oulad_week_0k.json` (|ρ| ≥ 0,9).

## 4. Catálogo de features — OULAD

### Engajamento digital

| Feature | Descrição | Fórmula / método | Origem | Disponível a partir de | Cortes | Tipo | Justificativa |
|---|---|---|---|---|---|---|---|
| `eng_total_clicks` | Total de cliques no AVA até o corte (inclui antes do início). | Σ sum_click, date ≤ corte | studentVle | cada clique, no seu dia | S2, S4, S6 | count | Volume de engajamento; separação 0,24–0,27 na Fase 02. |
| `eng_any_activity` | 1 se houve ao menos um clique até o corte. | 1[eng_total_clicks > 0] | studentVle | cada clique | S2, S4, S6 | binary | Distingue 'sem atividade' de 'pouca atividade'; 8% dos Withdrawn sem cliques em S2. |
| `eng_active_days` | Dias distintos com atividade até o corte. | nº de dias distintos com clique, date ≤ corte | studentVle | cada clique | S2, S4, S6 | count | Regularidade; proxy de frequência no EAD (não é presença). |
| `eng_clicks_per_active_day` | Média de cliques por dia ativo. | eng_total_clicks / eng_active_days (0 se nenhum dia ativo) | studentVle | cada clique | S2, S4, S6 | numeric | Intensidade por sessão, independente da regularidade. |
| `eng_clicks_per_elapsed_day` | Cliques por dia decorrido da apresentação. | Σ sum_click (0 ≤ date ≤ corte) / (corte + 1) | studentVle | cada clique | S2, S4, S6 | numeric | Normaliza pelo tempo decorrido, comparável entre cortes. |
| `eng_clicks_pre_start` | Cliques antes do dia 0. | Σ sum_click, date < 0 | studentVle | antes do início | S2, S4, S6 | count | Preparação prévia; separação 0,21 em S2. |
| `eng_active_days_pre_start` | Dias ativos antes do dia 0. | nº de dias distintos com clique, date < 0 | studentVle | antes do início | S2, S4, S6 | count | Complementa o volume pré-início com regularidade. |
| `eng_active_weeks` | Semanas (1..k) com ao menos um clique. | Σ_{w=1..k} 1[cliques_w > 0] | studentVle | fim de cada semana | S2, S4, S6 | count | Persistência semanal. |
| `eng_inactive_weeks` | Semanas (1..k) sem nenhum clique. | k − eng_active_weeks | studentVle | fim de cada semana | S2, S4, S6 | count | Inatividade semanal cresce de 41% para 65% antes do cancelamento (Fase 02). |
| `eng_active_weeks_ratio` | Percentual de semanas com atividade. | eng_active_weeks / k | studentVle | fim de cada semana | S2, S4, S6 | numeric | Comparável entre cortes com k diferente. |
| `eng_days_since_last_activity` | Dias desde o último clique (recência). | corte − último dia com clique ≤ corte; se nunca clicou: corte + 1 | studentVle | cada clique | S2, S4, S6 | numeric | Recência: separação 0,21–0,27. Valor para 'nunca clicou' é determinístico e acompanhado de eng_any_activity. |
| `eng_first_activity_day` | Dia do primeiro clique (negativo = antes do início). | mín(date) ≤ corte; se nunca clicou: corte + 1 | studentVle | cada clique | S2, S4, S6 | numeric | Início tardio do engajamento como sinal de risco. |
| `eng_distinct_sites` | Materiais distintos acessados. | nº de id_site distintos, date ≤ corte | studentVle | cada clique | S2, S4, S6 | count | Amplitude de exploração do curso. |
| `eng_distinct_activity_types` | Tipos de material distintos acessados. | nº de activity_type distintos, date ≤ corte | studentVle + vle | cada clique | S2, S4, S6 | count | Diversidade de atividades. |
| `eng_activity_type_entropy` | Entropia de Shannon dos cliques por tipo de material. | −Σ p_t log2 p_t, p_t = cliques do tipo t / total (0 se sem cliques) | studentVle + vle | cada clique | S2, S4, S6 | numeric | Diversidade ponderada pelo volume. |
| `eng_clicks_last_14_days` | Cliques nos 14 dias até o corte. | Σ sum_click, corte − 13 ≤ date ≤ corte | studentVle | cada clique | S4, S6 | count | Engajamento recente. Em S2 seria idêntico ao total do período → só S4/S6. |
| `eng_active_days_last_14_days` | Dias ativos nos 14 dias até o corte. | nº de dias com clique, corte − 13 ≤ date ≤ corte | studentVle | cada clique | S4, S6 | count | Frequência de acesso nas últimas duas semanas. |
| `eng_weekly_clicks_std` | Desvio-padrão dos cliques semanais (semanas 1..k). | desvio-padrão populacional de cliques_w, w = 1..k (zeros incluídos) | studentVle | fim de cada semana | S2, S4, S6 | numeric | Variação entre semanas. |
| `eng_weekly_clicks_cv` | Coeficiente de variação dos cliques semanais. | eng_weekly_clicks_std / média(cliques_w) (0 se média 0) | studentVle | fim de cada semana | S2, S4, S6 | numeric | Irregularidade relativa ao próprio volume. |
| `eng_weekly_trend_slope` | Inclinação (MQO) dos cliques semanais ao longo de 1..k. | cov(w, cliques_w) / var(w), w = 1..k | studentVle | fim de cada semana | S2, S4, S6 | numeric | Tendência de crescimento/redução: desengajamento progressivo. |
| `eng_last_week_change` | Variação de cliques da semana k−1 para a semana k. | cliques_k − cliques_{k−1} | studentVle | fim da semana k | S4, S6 | numeric | Queda abrupta recente. Em S2 (k = 2) é algebricamente idêntica a eng_weekly_trend_slope → só S4/S6. |
| `eng_clicks_pct_in_presentation` | Percentil dos cliques acumulados entre os estudantes elegíveis da mesma apresentação no corte. | rank percentual de eng_total_clicks dentro de (módulo, apresentação, corte) | studentVle | no corte (usa só dados ≤ corte de colegas elegíveis; nenhum rótulo) | S2, S4, S6 | numeric | Engajamento relativo ao contexto do módulo (CCC ≠ GGG). |
| `eng_clicks_week_01` | Cliques na semana 1 (dias 0–6). | Σ sum_click, 0 ≤ date ≤ 6 | studentVle | fim da semana 1 | S2, S4, S6 | count | Trajetória semanal completa até o corte. |
| `eng_clicks_week_02` | Cliques na semana 2 (dias 7–13). | Σ sum_click, 7 ≤ date ≤ 13 | studentVle | fim da semana 2 | S2, S4, S6 | count | Trajetória semanal completa até o corte. |
| `eng_clicks_week_03` | Cliques na semana 3 (dias 14–20). | Σ sum_click, 14 ≤ date ≤ 20 | studentVle | fim da semana 3 | S4, S6 | count | Trajetória semanal completa até o corte. |
| `eng_clicks_week_04` | Cliques na semana 4 (dias 21–27). | Σ sum_click, 21 ≤ date ≤ 27 | studentVle | fim da semana 4 | S4, S6 | count | Trajetória semanal completa até o corte. |
| `eng_clicks_week_05` | Cliques na semana 5 (dias 28–34). | Σ sum_click, 28 ≤ date ≤ 34 | studentVle | fim da semana 5 | S6 | count | Trajetória semanal completa até o corte. |
| `eng_clicks_week_06` | Cliques na semana 6 (dias 35–41). | Σ sum_click, 35 ≤ date ≤ 41 | studentVle | fim da semana 6 | S6 | count | Trajetória semanal completa até o corte. |

### Atividades e avaliações (sem notas)

| Feature | Descrição | Fórmula / método | Origem | Disponível a partir de | Cortes | Tipo | Justificativa |
|---|---|---|---|---|---|---|---|
| `asm_n_due` | Avaliações não-finais com prazo ≤ corte (calendário). | nº de assessments (TMA/CMA) com date ≤ corte | assessments | calendário publicado | S4, S6 | count | OULAD não tem data de abertura: 'disponibilizada' é aproximada por 'prazo vencido'. Em S2 só 4,6% das apresentações têm prazo vencido → excluída de S2. |
| `asm_n_submitted` | Avaliações não-finais entregues até o corte (não-banked). | nº de entregas com date_submitted ≤ corte, is_banked = 0 | studentAssessment | dia da entrega | S4, S6 | count | Participação; inclui entregas antecipadas. |
| `asm_n_due_submitted` | Avaliações com prazo ≤ corte e entregues até o corte. | nº de (prazo ≤ corte) ∧ (date_submitted ≤ corte) | studentAssessment | dia da entrega | S4, S6 | count | Base da taxa de entrega. |
| `asm_submission_rate` | Percentual de avaliações vencidas que foram entregues. | asm_n_due_submitted / asm_n_due (vazio se asm_n_due = 0) | studentAssessment | dia da entrega | S4, S6 | numeric | Indefinida sem prazo vencido: deixada ausente para imputação ajustada só no treino. |
| `asm_n_missing` | Avaliações vencidas não entregues até o corte. | asm_n_due − asm_n_due_submitted | studentAssessment | prazo | S4, S6 | count | 22,6% dos Withdrawn não entregam a 1ª avaliação (Pass: 1%). |
| `asm_missed_weight` | Soma dos pesos (%) das avaliações vencidas não entregues. | Σ weight das avaliações em asm_n_missing | assessments + studentAssessment | prazo | S4, S6 | numeric | Quanto da nota já foi perdido por não entrega. |
| `asm_n_on_time` | Entregas até o corte feitas dentro do prazo. | nº de entregas com date_submitted ≤ min(prazo, corte) | studentAssessment | dia da entrega | S4, S6 | count | Pontualidade. |
| `asm_n_late` | Entregas até o corte feitas após o prazo. | nº de entregas com prazo < date_submitted ≤ corte | studentAssessment | dia da entrega | S4, S6 | count | Atraso como sinal de dificuldade. |
| `asm_mean_days_before_deadline` | Média de dias entre a entrega e o prazo (positivo = antecipada). | média(prazo − date_submitted) das entregas ≤ corte (vazio se nenhuma) | studentAssessment | dia da entrega | S4, S6 | numeric | Substitui 'dias entre disponibilização e submissão': a data de abertura não existe na OULAD. |
| `asm_first_due_passed` | 1 se o prazo da 1ª avaliação não-final já venceu no corte. | 1[prazo da 1ª avaliação ≤ corte] | assessments | calendário | S4, S6 | binary | Contexto para interpretar asm_first_submitted. |
| `asm_first_submitted` | 1 se a 1ª avaliação não-final foi entregue até o corte. | 1[date_submitted(1ª avaliação) ≤ corte] | studentAssessment | dia da entrega | S4, S6 | binary | Participação nas primeiras avaliações. |

### Contexto do módulo

| Feature | Descrição | Fórmula / método | Origem | Disponível a partir de | Cortes | Tipo | Justificativa |
|---|---|---|---|---|---|---|---|
| `ctx_code_module` | Módulo. | code_module | courses | antes do início | S2, S4, S6 | nominal | Taxa de Withdrawn de 11,5% a 44,5% entre módulos. Risco: aprender a turma. |
| `ctx_presentation_start` | Mês de início da apresentação (B = fevereiro, J = outubro). | último caractere de code_presentation | courses | antes do início | S2, S4, S6 | nominal | Sazonalidade sem expor o ano (o ano será usado para validação temporal). |
| `ctx_module_length` | Duração planejada da apresentação (dias). | module_presentation_length | courses | antes do início | S2, S4, S6 | numeric | Planejada e publicada. |
| `ctx_n_planned_assessments` | Nº de avaliações não-finais planejadas. | nº de TMA/CMA da apresentação | assessments | antes do início | S2, S4, S6 | count | Carga avaliativa planejada. |

### Características de ingresso e histórico

| Feature | Descrição | Fórmula / método | Origem | Disponível a partir de | Cortes | Tipo | Justificativa |
|---|---|---|---|---|---|---|---|
| `adm_gender` ⚠️ sensível | Gênero declarado. | gender | studentInfo | matrícula | S2, S4, S6 | nominal | Associação fraca (V = 0,02). Sensível: avaliar equidade. |
| `adm_region` | Região de residência. | region | studentInfo | matrícula | S2, S4, S6 | nominal | V = 0,07. |
| `adm_highest_education` | Escolaridade prévia (ordinal 0–4). | No Formal quals=0 … Post Graduate Qualification=4 | studentInfo | matrícula | S2, S4, S6 | ordinal | 42,9% × 23,6% de Withdrawn entre extremos. |
| `adm_imd_band` | Faixa do índice de privação (ordinal 0–9; 0 = mais privado). | 0-10%=0 … 90-100%=9; ausente mantido ausente | studentInfo | matrícula | S2, S4, S6 | ordinal | 37,2% × 25,9%. 1.111 ausentes: imputação ajustada só no treino. |
| `adm_age_band` | Faixa etária (ordinal 0–2). | 0-35=0, 35-55=1, 55<=2 | studentInfo | matrícula | S2, S4, S6 | ordinal | V = 0,06. |
| `adm_disability` ⚠️ sensível | Declarou deficiência. | 1[disability = Y] | studentInfo | matrícula | S2, S4, S6 | binary | 39,4% × 30,3%. Sensível: avaliar equidade. |
| `adm_num_prev_attempts` | Tentativas anteriores no módulo. | num_of_prev_attempts | studentInfo | matrícula | S2, S4, S6 | count | Histórico anterior ao início. |
| `adm_registration_lead_days` | Antecedência da matrícula (dias antes do início). | −date_registration (negativo = matrícula após o início); ausente mantido ausente | studentRegistration | matrícula | S2, S4, S6 | numeric | ≤ −90 dias: 38,2% × após o início: 20,9%. Sem indicador de ausência (ver quarentena). |

### Em quarentena (não produzidas nos conjuntos principais)

| Feature | Motivo |
|---|---|
| `studied_credits` | Data de registro não documentada (pode mudar no ano). Forte associação (51,3% × 25,4%) — quarentena até comprovação. |
| `asm_mean_score` | Data de divulgação inexistente: não pode ser assumida conhecida na entrega. |
| `asm_banked_results` | Semântica de data não documentada. |
| `adm_registration_missing` | 86,7% dos 45 casos são Withdrawn: ausência possivelmente causada pelo cancelamento. |
| `ctx_presentation_year` | Mantido como coluna de auditoria para validação temporal; como feature impediria generalizar para anos futuros. |

## 5. UCI 697 — conjunto de ingresso

18 features, todas documentadas pelos autores como **conhecidas no ingresso**
(`catalog.UCI_ADMISSION_FEATURES`). Tipos registrados nos metadados para codificação
futura ajustada no treino.

| Feature | Tipo | Justificativa |
|---|---|---|
| `Marital status`, `Application mode`, `Course`, `Previous qualification`, `Nacionality`, `Mother's qualification`, `Father's qualification`, `Mother's occupation`, `Father's occupation` | nominal (códigos inteiros sem ordem) | Ingresso. Codificar como categoria, nunca como número |
| `Daytime/evening attendance`, `Displaced`, `Educational special needs`, `Gender` ⚠️, `International` | binária | Ingresso. `Gender` sensível |
| `Application order`, `Previous qualification (grade)`, `Admission grade`, `Age at enrollment` | numérica | Ingresso |

**Excluídas por disponibilidade não comprovada (quarentena):** `Scholarship holder`
(pedida na especificação, mas a data de coleta é desconhecida e bolsas mudam durante o
curso), `Debtor`, `Tuition fees up to date`, `Curricular units 1st sem (enrolled/credited)`,
`Unemployment rate`, `Inflation rate`, `GDP` (as três macro ficam como coluna de auditoria
`cohort_proxy`, útil para validação por coorte).

**Proibidas:** todas as variáveis de avaliação/aprovação/nota do 1º e 2º semestres,
inscrições do 2º semestre e `Target`. O construtor recusa a lista de features se alguma
delas aparecer (teste `test_uci_guard_rejects_forbidden_features`).

Redundância conhecida: `Nacionality` × `International` (ρ = 1,0) — mantidas e registradas.
