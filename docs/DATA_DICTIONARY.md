# Dicionário de Dados Preliminar — EduGuard-IA

> Escopo: datasets **efetivamente baixados** na Fase 01: `uci_dropout_697` e
> `oulad_349`. Estatísticas completas em `research/inspection/*_profile.json`.

## Legenda da coluna "Disponível em"

| Código | Momento em que o valor é conhecido |
|---|---|
| **M** | Matrícula / ingresso (antes do 1º semestre) |
| **F1** | Fim do 1º semestre |
| **F2** | Fim do 2º semestre |
| **?** | Momento de coleta **não documentado** — risco de vazamento |
| **C** | Contexto macroeconômico (provavelmente do ano de ingresso — inferência) |
| **Y** | Desfecho (alvo) — ao final da duração normal do curso |

## uci_dropout_697 — `data.csv`

- Arquivo: `data/raw/uci_dropout_697/extracted/data.csv`
- Separador `;`, UTF-8 **com BOM**; 4.424 linhas × 37 colunas.
- 0 valores ausentes; 0 linhas duplicadas (nem considerando só as features).
- Inconsistência: o cabeçalho `Daytime/evening attendance` contém um **TAB** ao final
  (`"Daytime/evening attendance\t"`). O arquivo original não foi alterado; o
  carregamento remove espaços em memória.
- Inconsistência de metadado: a UCI declara as médias semestrais como `Integer`,
  mas os valores são contínuos (ex.: 18,875).

| # | Coluna original | Tipo | Faixa observada | Descrição (UCI) | Disponível em |
|---|---|---|---|---|---|
| 1 | Marital status | categórica (int) | 1–6 | 1 solteiro, 2 casado, 3 viúvo, 4 divorciado, 5 união de fato, 6 separado | M |
| 2 | Application mode | categórica (int) | 18 códigos (1–57) | Modalidade de candidatura (fase, contingente, transferência, >23 anos…) | M |
| 3 | Application order | ordinal | 0–9 | Ordem de preferência do curso (0 = 1ª opção) | M |
| 4 | Course | categórica (int) | 17 cursos | 33 Biofuel Prod. Tech.; 171 Animation & Multimedia Design; 8014 Social Service (noturno); 9003 Agronomy; 9070 Communication Design; 9085 Veterinary Nursing; 9119 Informatics Eng.; 9130 Equinculture; 9147 Management; 9238 Social Service; 9254 Tourism; 9500 Nursing; 9556 Oral Hygiene; 9670 Advertising & Marketing Mgmt; 9773 Journalism & Communication; 9853 Basic Education; 9991 Management (noturno) | M |
| 5 | Daytime/evening attendance | binária | 0/1 | 1 diurno, 0 noturno | M |
| 6 | Previous qualification | categórica (int) | 17 códigos | Escolaridade anterior | M |
| 7 | Previous qualification (grade) | contínua | 95–190 | Nota da qualificação anterior (0–200) | M |
| 8 | Nacionality | categórica (int) | 21 códigos | Nacionalidade (41 = brasileira) — grafia original "Nacionality" | M |
| 9 | Mother's qualification | categórica (int) | 29 códigos | Escolaridade da mãe | M |
| 10 | Father's qualification | categórica (int) | 34 códigos | Escolaridade do pai | M |
| 11 | Mother's occupation | categórica (int) | 32 códigos | Ocupação da mãe | M |
| 12 | Father's occupation | categórica (int) | 46 códigos | Ocupação do pai | M |
| 13 | Admission grade | contínua | 95–190 | Nota de admissão (0–200) | M |
| 14 | Displaced | binária | 0/1 | Deslocado da residência de origem | M |
| 15 | Educational special needs | binária | 0/1 | Necessidades educacionais especiais (1,15% = 1) | M |
| 16 | Debtor | binária | 0/1 | Devedor | **?** |
| 17 | Tuition fees up to date | binária | 0/1 | Propinas/mensalidades em dia | **?** |
| 18 | Gender | binária | 0/1 | 1 masculino, 0 feminino | M |
| 19 | Scholarship holder | binária | 0/1 | Bolsista | **?** |
| 20 | Age at enrollment | inteira | 17–70 | Idade no ingresso | M |
| 21 | International | binária | 0/1 | Estudante internacional | M |
| 22 | Curricular units 1st sem (credited) | inteira | 0–20 | UCs creditadas (aproveitamento) no 1º sem. | M/F1 * |
| 23 | Curricular units 1st sem (enrolled) | inteira | 0–26 | UCs inscritas no 1º sem. | início do 1º sem. * |
| 24 | Curricular units 1st sem (evaluations) | inteira | 0–45 | Nº de avaliações no 1º sem. | **F1** |
| 25 | Curricular units 1st sem (approved) | inteira | 0–26 | UCs aprovadas no 1º sem. | **F1** |
| 26 | Curricular units 1st sem (grade) | contínua | 0–18,875 | Média do 1º sem. (0–20) | **F1** |
| 27 | Curricular units 1st sem (without evaluations) | inteira | 0–12 | UCs sem avaliação no 1º sem. | **F1** |
| 28–33 | Curricular units 2nd sem (credited / enrolled / evaluations / approved / grade / without evaluations) | idem | idem | Mesmo conteúdo, 2º semestre | **F2** (enrolled: início do 2º sem.) |
| 34 | Unemployment rate | contínua | 7,6–16,2 | Taxa de desemprego (%) | C |
| 35 | Inflation rate | contínua | −0,8–3,7 | Inflação (%) | C |
| 36 | GDP | contínua | −4,06–3,51 | PIB (variação) | C |
| 37 | **Target** | categórica | 3 classes | `Dropout` 1.421 · `Enrolled` 794 · `Graduate` 2.209 | **Y** |

\* "enrolled"/"credited" provavelmente são conhecidos no início do semestre
(inscrição), mas a UCI **não documenta** o instante da extração. Tratar com cautela.

### Observações da inspeção (sem feature engineering)

- `Curricular units 1st sem (enrolled) = 0` em 180 estudantes (77 Dropout,
  28 Enrolled, 75 Graduate) — possíveis casos de creditação total; investigar.
- Nenhum caso de `approved > enrolled` e nenhum `grade = 0` com aprovações > 0.
- **169** estudantes com UCs inscritas e **0 avaliações no 1º semestre**: todos `Dropout`.
- `Curricular units 2nd sem (approved) = 0`: 51,2% dos Dropout × 3,4% dos Graduate.
- `Tuition fees up to date = 0`: 32,2% dos Dropout × 1,3% dos Graduate.
- `Debtor = 1`: 22,0% dos Dropout × 4,6% dos Graduate.
- 10 combinações únicas de (desemprego, inflação, PIB) — provável proxy de coorte;
  pode viabilizar validação temporal **se** a correspondência com o ano for confirmada.
- Desbalanceamento moderado: Dropout = 32,1%.

### Mapeamento provisório para o rótulo do EduGuard

| Target UCI | Rótulo EduGuard | Status |
|---|---|---|
| Dropout | EVADIU = 1 | direto |
| Graduate | NÃO EVADIU = 0 | direto |
| Enrolled | ? | **pendente** — desfecho censurado (ainda matriculado após a duração normal) |

---

## oulad_349 — Open University Learning Analytics Dataset

- Pasta: `data/raw/oulad_349/extracted/` (7 CSVs + `OULAD.names`), separador `,`,
  aspas em todos os campos, quebras de linha CRLF.
- **Valores ausentes são codificados como `"?"`** — ler com `na_values="?"`.
- Chave de estudante-no-módulo: (`code_module`, `code_presentation`, `id_student`).
  O mesmo `id_student` aparece em várias apresentações (28.785 IDs × 32.593 linhas).
- **Todas as datas são inteiros em dias relativos ao início da apresentação (dia 0).**
  `code_presentation` = ano + `B` (início em fevereiro) ou `J` (início em outubro).

Legenda "Disponível em": **M** = antes do início/matrícula · **D** = no dia indicado
pela própria coluna de data (usar só se data ≤ corte) · **Y** = desfecho · **?** = não documentado.

### courses.csv — 22 linhas

| Coluna | Tipo | Descrição | Disponível em |
|---|---|---|---|
| code_module | categórica | Módulo (AAA…GGG, 7 valores) | M |
| code_presentation | categórica | Apresentação (2013B, 2013J, 2014B, 2014J) | M |
| module_presentation_length | inteiro | Duração em dias (234–269) | M |

### assessments.csv — 206 linhas

| Coluna | Tipo | Descrição | Disponível em |
|---|---|---|---|
| code_module, code_presentation | categórica | Chave do módulo | M |
| id_assessment | inteiro | ID da avaliação | M |
| assessment_type | categórica | TMA 106 (tutor), CMA 76 (computador), Exam 24 | M |
| date | inteiro (11 ausentes) | Prazo final de entrega (dia) — 12 a 261 | M (calendário publicado) |
| weight | numérico | Peso (%) — 0 a 100 | M |

### vle.csv — 6.364 linhas

| Coluna | Tipo | Descrição | Disponível em |
|---|---|---|---|
| id_site | inteiro | ID do recurso no AVA | M |
| code_module, code_presentation | categórica | Chave do módulo | M |
| activity_type | categórica (20) | Tipo do recurso (ex.: resource, forumng, quiz…) | M |
| week_from, week_to | inteiro (5.243 ausentes = 82%) | Semanas planejadas de uso | M |

### studentInfo.csv — 32.593 linhas

| Coluna | Tipo | Descrição | Disponível em |
|---|---|---|---|
| code_module, code_presentation, id_student | chave | — | M |
| gender | categórica | M/F | M |
| region | categórica (13) | Região do Reino Unido | M |
| highest_education | categórica (5) | Escolaridade prévia | M |
| imd_band | ordinal (10; **1.111 ausentes**) | Faixa do índice de privação múltipla da região | M |
| age_band | ordinal (3) | 0-35, 35-55, 55<= | M |
| num_of_prev_attempts | inteiro (0–6) | Tentativas anteriores no módulo | M |
| studied_credits | inteiro (30–655) | Créditos cursados no período | M ? (pode mudar ao longo do ano) |
| disability | binária | Declarou deficiência | M |
| **final_result** | categórica | **Pass 12.361 · Withdrawn 10.156 · Fail 7.052 · Distinction 3.024** | **Y** |

### studentRegistration.csv — 32.593 linhas

| Coluna | Tipo | Descrição | Disponível em |
|---|---|---|---|
| chave | — | — | M |
| date_registration | inteiro (45 ausentes) | Dia da matrícula (−322 a 167) | D |
| date_unregistration | inteiro (22.521 ausentes = não cancelou) | Dia do cancelamento (−365 a 444) | **Y** (define o evento e quem sai da população em cada corte) |

### studentAssessment.csv — 173.912 linhas

| Coluna | Tipo | Descrição | Disponível em |
|---|---|---|---|
| id_assessment, id_student | chave | — | — |
| date_submitted | inteiro (−11 a 608) | Dia da entrega | D |
| is_banked | binária | Nota aproveitada de apresentação anterior | D ? |
| score | numérico 0–100 (173 ausentes) | Nota | D ? (data de divulgação **não existe**) |

### studentVle.csv — 10.655.280 linhas (453,8 MB)

| Coluna | Tipo | Descrição | Disponível em |
|---|---|---|---|
| chave + id_site | — | Estudante × recurso | — |
| date | inteiro (−25 a 269) | Dia da interação; 688.988 linhas antes do dia 0 | D |
| sum_click | inteiro | Cliques naquele dia naquele recurso | D |

### Mapeamento provisório para o rótulo do EduGuard

| final_result | Rótulo EduGuard | Status |
|---|---|---|
| Withdrawn | EVADIU = 1 | direto (cancelamento do **módulo**) |
| Pass, Distinction | NÃO EVADIU = 0 | direto |
| Fail | ? | **pendente** — concluiu sem aprovação; não abandonou formalmente |
