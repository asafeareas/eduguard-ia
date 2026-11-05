# Matriz de Correspondência OULAD → Contexto Brasileiro

> Fase 05 · 2026-11-05. Fonte única: `src/validation/feature_mapping.py` (tabela gerada a partir dela).
> **Nenhum dado brasileiro foi acessado nesta fase.** O que se afirma sobre a UFRN vem apenas
> dos cabeçalhos e metadados públicos lidos na Fase 01 (2026-11-05).
>
> Critério conservador: uma feature só é "direta" se a **mesma definição** puder ser calculada a
> partir de registros acadêmicos brasileiros. Se o conceito é parecido mas a definição, a escala
> ou a plataforma mudam, ela é "exige transformação", e o modelo treinado na OULAD **não** pode
> recebê-la como se fosse a mesma variável.

## 1. Resumo

| | S2 (30 preditores v2) | S4 (46) | S6 (48) |
|---|---|---|---|
| ✅ direta | 1 | 1 | 1 |
| 🔄 exige transformação | 26 | 42 | 44 |
| ❌ sem correspondência | 0 | 0 | 0 |
| ⛔ não usar | 3 | 3 | 3 |

**Transferência direta do modelo v2: impossível**, seja com os dados abertos da UFRN
(`direct_transfer_possible = False`), seja com todos os campos opcionais do
contrato de dados (`False`). Só `adm_age_band` é direta. As features mais importantes do modelo
(avaliações vencidas não entregues e engajamento no AVA) exigem logs do AVA e entregas datadas, que
não existem nos dados abertos da UFRN e, mesmo que existam internamente, mudam de semântica
entre plataformas.

## 2. Matriz completa


### Engajamento digital

| Feature OULAD | Status | Equivalente brasileiro esperado | Campos necessários | Observações |
|---|---|---|---|---|
| `eng_total_clicks` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_any_activity` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_active_days` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_clicks_per_active_day` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_clicks_per_elapsed_day` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_active_weeks` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_inactive_weeks` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_active_weeks_ratio` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_days_since_last_activity` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_first_activity_day` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_clicks_last_14_days` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_active_days_last_14_days` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_weekly_clicks_std` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_weekly_clicks_cv` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_weekly_trend_slope` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_last_week_change` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_clicks_pct_in_presentation` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_clicks_week_01` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_clicks_week_02` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_clicks_week_03` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_clicks_week_04` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_clicks_week_05` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_clicks_week_06` | 🔄 exige transformação | Acessos ao AVA da disciplina até o corte | `ava_log_timestamp`, `ava_log_student` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_clicks_pre_start` | 🔄 exige transformação | Acessos ao AVA antes do início das aulas | `ava_log_timestamp`, `semester_start_date` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. No Brasil o AVA costuma abrir junto com as aulas. |
| `eng_active_days_pre_start` | 🔄 exige transformação | Dias com acesso antes do início das aulas | `ava_log_timestamp`, `semester_start_date` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_distinct_sites` | 🔄 exige transformação | Recursos distintos acessados no AVA | `ava_log_resource` | Exige logs do AVA com data/hora por estudante (ex.: Moodle ou turma virtual). Não existem nos dados abertos da UFRN verificados na Fase 01. Mesmo com logs, 'clique' tem semântica diferente entre plataformas → redefinição e recalibração obrigatórias. |
| `eng_distinct_activity_types` | 🔄 exige transformação | Tipos de recurso distintos (taxonomia do AVA local) | `ava_log_resource_type` | Os 20 tipos da OU (oucontent, forumng…) não correspondem aos módulos do Moodle/SIGAA; exige tabela de correspondência validada. |
| `eng_activity_type_entropy` | 🔄 exige transformação | Entropia por tipo de recurso | `ava_log_resource_type` | Depende da taxonomia; não comparável sem harmonização. |

### Atividades e avaliações

| Feature OULAD | Status | Equivalente brasileiro esperado | Campos necessários | Observações |
|---|---|---|---|---|
| `asm_n_due` | 🔄 exige transformação | Atividades avaliativas com prazo e entrega datada | `activity_due_date`, `submission_timestamp` | Exige calendário de atividades (prazos) e data/hora de entrega. Os dados abertos da UFRN têm nota e faltas por unidade, sem datas (Fase 01). Só seria computável a partir de atividades entregues no AVA. |
| `asm_n_submitted` | 🔄 exige transformação | Atividades avaliativas com prazo e entrega datada | `activity_due_date`, `submission_timestamp` | Exige calendário de atividades (prazos) e data/hora de entrega. Os dados abertos da UFRN têm nota e faltas por unidade, sem datas (Fase 01). Só seria computável a partir de atividades entregues no AVA. |
| `asm_n_due_submitted` | 🔄 exige transformação | Atividades avaliativas com prazo e entrega datada | `activity_due_date`, `submission_timestamp` | Exige calendário de atividades (prazos) e data/hora de entrega. Os dados abertos da UFRN têm nota e faltas por unidade, sem datas (Fase 01). Só seria computável a partir de atividades entregues no AVA. |
| `asm_submission_rate` | 🔄 exige transformação | Atividades avaliativas com prazo e entrega datada | `activity_due_date`, `submission_timestamp` | Exige calendário de atividades (prazos) e data/hora de entrega. Os dados abertos da UFRN têm nota e faltas por unidade, sem datas (Fase 01). Só seria computável a partir de atividades entregues no AVA. |
| `asm_n_missing` | 🔄 exige transformação | Atividades avaliativas com prazo e entrega datada | `activity_due_date`, `submission_timestamp` | Exige calendário de atividades (prazos) e data/hora de entrega. Os dados abertos da UFRN têm nota e faltas por unidade, sem datas (Fase 01). Só seria computável a partir de atividades entregues no AVA. |
| `asm_n_on_time` | 🔄 exige transformação | Atividades avaliativas com prazo e entrega datada | `activity_due_date`, `submission_timestamp` | Exige calendário de atividades (prazos) e data/hora de entrega. Os dados abertos da UFRN têm nota e faltas por unidade, sem datas (Fase 01). Só seria computável a partir de atividades entregues no AVA. |
| `asm_n_late` | 🔄 exige transformação | Atividades avaliativas com prazo e entrega datada | `activity_due_date`, `submission_timestamp` | Exige calendário de atividades (prazos) e data/hora de entrega. Os dados abertos da UFRN têm nota e faltas por unidade, sem datas (Fase 01). Só seria computável a partir de atividades entregues no AVA. |
| `asm_mean_days_before_deadline` | 🔄 exige transformação | Atividades avaliativas com prazo e entrega datada | `activity_due_date`, `submission_timestamp` | Exige calendário de atividades (prazos) e data/hora de entrega. Os dados abertos da UFRN têm nota e faltas por unidade, sem datas (Fase 01). Só seria computável a partir de atividades entregues no AVA. |
| `asm_first_due_passed` | 🔄 exige transformação | Atividades avaliativas com prazo e entrega datada | `activity_due_date`, `submission_timestamp` | Exige calendário de atividades (prazos) e data/hora de entrega. Os dados abertos da UFRN têm nota e faltas por unidade, sem datas (Fase 01). Só seria computável a partir de atividades entregues no AVA. |
| `asm_first_submitted` | 🔄 exige transformação | Atividades avaliativas com prazo e entrega datada | `activity_due_date`, `submission_timestamp` | Exige calendário de atividades (prazos) e data/hora de entrega. Os dados abertos da UFRN têm nota e faltas por unidade, sem datas (Fase 01). Só seria computável a partir de atividades entregues no AVA. |
| `asm_missed_weight` | 🔄 exige transformação | Peso das atividades vencidas não entregues | `activity_due_date`, `submission_timestamp`, `activity_weight` | Exige calendário de atividades (prazos) e data/hora de entrega. Os dados abertos da UFRN têm nota e faltas por unidade, sem datas (Fase 01). Só seria computável a partir de atividades entregues no AVA. Pesos raramente são registrados no sistema acadêmico. |

### Contexto do módulo

| Feature OULAD | Status | Equivalente brasileiro esperado | Campos necessários | Observações |
|---|---|---|---|---|
| `ctx_code_module` | ⛔ não usar | Nenhuma (identidade dos 7 módulos da OU) | — | Categorias AAA–GGG não existem fora da OU. O one-hot ignoraria valores novos, transformando silenciosamente todos os cursos brasileiros na categoria de referência. Não mapear. |
| `ctx_presentation_start` | ⛔ não usar | Nenhuma (fevereiro/outubro do calendário britânico) | — | Semestres brasileiros (1º/2º) não equivalem às apresentações B/J. |
| `ctx_module_length` | 🔄 exige transformação | Duração do período letivo (dias) | `semester_start_date`, `semester_end_date` | Módulos da OU têm de 234 a 269 dias; semestres brasileiros, cerca de 100 a 120 → fora da faixa de treino. |
| `ctx_n_planned_assessments` | 🔄 exige transformação | Nº de avaliações/unidades previstas no plano da disciplina | `planned_assessments` | A UFRN registra 'unidade' (1–3) nas matrículas em componentes; não é o mesmo que TMA/CMA. |

### Ingresso, histórico e auditoria

| Feature OULAD | Status | Equivalente brasileiro esperado | Campos necessários | Observações |
|---|---|---|---|---|
| `adm_region` | ⛔ não usar | Nenhuma (13 regiões do Reino Unido) | — | Sem correspondência geográfica. |
| `adm_highest_education` | 🔄 exige transformação | Escolaridade prévia (ensino médio, superior incompleto…) | `prior_education` | As qualificações britânicas (A Level, HE…) exigem tabela de equivalência validada; a ordem 0–4 não é garantida. |
| `adm_age_band` | ✅ direta | Faixa etária no início do período (0–35, 35–55, 55+) | `birth_year` | Mesma definição. Minimizar: enviar só a faixa, nunca a data de nascimento. |
| `adm_num_prev_attempts` | 🔄 exige transformação | Matrículas anteriores na mesma disciplina | `previous_enrollments_same_component` | Definição de 'tentativa' varia (trancamento conta?). |
| `adm_registration_lead_days` | 🔄 exige transformação | Antecedência da matrícula na disciplina | `enrollment_date`, `semester_start_date` | No Brasil a matrícula segue o calendário institucional (janelas fixas): a variação tem outro significado. |
| `adm_gender` | ⛔ não usar | Sexo/gênero (apenas auditoria) | `gender` | Q16: nunca preditor. |
| `adm_disability` | ⛔ não usar | Deficiência (apenas auditoria; dado sensível — LGPD art. 5º, II) | `disability` | Q16/Q18: nunca preditor; exige base legal específica (art. 11). |
| `adm_imd_band` | ⛔ não usar | Indicador socioeconômico (ex.: renda per capita, cota) — apenas auditoria | `socioeconomic_indicator` | Q17: removido do modelo principal; o IMD é britânico e sem equivalente direto. |

## 3. Informações brasileiras sem correspondência na OULAD

Servem para **retreinamento**, não para transferência.

| Informação | Descrição |
|---|---|
| `frequencia_faltas` | Faltas por unidade/total (UFRN: faltas_unidade, numero_total_faltas). Frequência é obrigatória no presencial (LDB, art. 47 §3º); o mínimo (usualmente 75%) vem do regulamento de cada instituição — verificar. |
| `notas_por_unidade` | Notas parciais por unidade (UFRN: nota, unidade) — sem data de lançamento nos dados abertos. |
| `situacao_componente` | Aprovado / reprovado por nota / por falta / cancelado / trancado (UFRN: descricao). |
| `historico_semestres_anteriores` | Desempenho em períodos anteriores do mesmo curso. |
| `forma_de_ingresso` | SiSU, transferência, cotas (sensível: tratar como auditoria). |

## 4. Campos públicos da UFRN verificados na Fase 01

- `matriculas-componentes`: `id_turma`, `discente`, `id_curso`, `unidade`, `nota`, `reposicao`, `faltas_unidade`, `media_final`, `numero_total_faltas`, `descricao`
- `situacoes-de-discentes`: `id_discente`, `ano_ingresso`, `periodo_ingresso`, `nivel_ensino`, `id_curso`, `curso`, `situacao`, `data_alteracao_situacao`, `ano_alteracao_situacao`, `periodo_alteracao_situacao`

Pontos importantes:
- Nenhuma dessas tabelas tem data de lançamento de notas ou faltas, então não há como montar cortes semanais.
- Os identificadores são *hashes*. Não foi verificado se `discente` (matrículas) e `id_discente` (situações) usam o mesmo hash (pendência Q10).
- O conjunto `discentes` da UFRN contém nomes e matrículas reais e **não deve ser usado** (Fase 01).
