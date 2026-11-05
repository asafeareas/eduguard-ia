# Candidatas a Features — Lista Preliminar

> Fase 02 · 2026-11-05. **Esta não é a seleção definitiva.** A lista reúne variáveis que
> podem ser calculadas sem vazamento em cada corte. A seleção, a engenharia e a
> validação ficam para a Fase 03, mediante autorização.
>
> Coluna "Corte": a partir de qual momento a variável faz sentido (S0 = matrícula/dia 0).
> "Evidência" = força de separação univariada observada na Fase 02 (Withdrawn × Pass/Distinction
> na OULAD; Dropout × Graduate na UCI), quando medida. É descritiva, não causal.

## 1. Desempenho acadêmico

| Candidata | Dataset | Origem | Corte | Classe | Evidência / nota |
|---|---|---|---|---|---|
| Nota média das avaliações entregues até o corte | OULAD | studentAssessment.score | S4+ | CONDICIONAL | 0,355 (S4), 0,388 (S6); cobertura 75–87%. Testar defasagem de divulgação |
| Nota na 1ª avaliação não-final | OULAD | studentAssessment | S4+ (depende do prazo) | CONDICIONAL | Mediana 70 (Withdrawn) × 78 (Pass) |
| Nº de avaliações entregues até o corte | OULAD | studentAssessment | S2+ | CONDICIONAL | 0,133 (S4) |
| Nº de avaliações vencidas e **não entregues** | OULAD | assessments + studentAssessment | S4+ | CONDICIONAL | 0,204 (S4); 22,6% dos Withdrawn não entregam a 1ª avaliação |
| Nº de entregas atrasadas | OULAD | date_submitted × date | S4+ | CONDICIONAL | Não medido isoladamente |
| Taxa de aprovação / média do 1º semestre | UCI | Curricular units 1st sem | **Só no 2º semestre** | PERMITIDA nesse cenário | 0,717 / 0,618 — mas com problema de população (ver TEMPORAL_FEASIBILITY §4) |

## 2. Frequência

| Candidata | Dataset | Corte | Classe | Nota |
|---|---|---|---|---|
| Faltas por unidade / total | **UFRN** (não baixado) | a avaliar | — | Única base encontrada com frequência real |
| Dias com atividade no AVA (proxy) | OULAD | S2+ | CONDICIONAL | 0,234–0,260. **Não é frequência presencial** |
| — | UCI | — | — | Não existe |

## 3. Engajamento digital (OULAD)

| Candidata | Corte | Classe | Evidência |
|---|---|---|---|
| Cliques acumulados até o corte | S2+ | CONDICIONAL | 0,272 (S2) / 0,244 (S4) / 0,255 (S6) |
| Dias ativos até o corte | S2+ | CONDICIONAL | 0,234 / 0,242 / 0,260 |
| Materiais distintos acessados | S2+ | CONDICIONAL | 0,219 / 0,180 / 0,172 |
| Cliques antes do início (dias < 0) | S0+ | CONDICIONAL | 0,207 (S2) |
| Cliques nos últimos 7 dias | S2+ | CONDICIONAL | 0,251 / 0,224 / 0,306 |
| Dias desde o último acesso (recência) | S2+ | CONDICIONAL | 0,208 / 0,223 / 0,273 |
| Tendência: últimos 7 dias × 7 dias anteriores | S2+ | CONDICIONAL | Já calculada (`clicks_prev_7_days`); não avaliada isoladamente |
| Cliques por tipo de material (fórum, quiz, conteúdo) | S2+ | CONDICIONAL | Diferenças de composição pequenas |
| Engajamento relativo ao módulo (percentil no módulo) | S2+ | CONDICIONAL | Normalizar só com dados ≤ corte e ajuste no treino |

## 4. Histórico acadêmico

| Candidata | Dataset | Corte | Classe | Evidência / nota |
|---|---|---|---|---|
| num_of_prev_attempts | OULAD | S0 | PERMITIDA | 30,6% (0 tentativas) → 36,6% (2) de Withdrawn |
| studied_credits | OULAD | S0 | CONDICIONAL | >120 créditos: 51,3% × ≤60: 25,4% — data de registro incerta |
| highest_education | OULAD | S0 | PERMITIDA | 42,9% (sem qualificação) × 23,6% (pós-graduação) |
| Previous qualification (+ nota) | UCI | S0 | PERMITIDA | Nota: 0,137 |
| Admission grade | UCI | S0 | PERMITIDA | 0,155 |
| Curricular units 1st sem (enrolled / credited) | UCI | início do sem. | CONDICIONAL | 0,272 (inscritas) |

## 5. Características de ingresso

| Candidata | Dataset | Classe | Evidência / nota |
|---|---|---|---|
| Idade (age_band / Age at enrollment) | ambos | PERMITIDA | UCI: 0,368 (melhor permitida); OULAD: V = 0,058 |
| imd_band (privação) | OULAD | PERMITIDA | 37,2% × 25,9% |
| Data de matrícula (antecedência) | OULAD | PERMITIDA | ≤ −90 dias: 38,2% × após o início: 20,9% |
| Application mode / order | UCI | PERMITIDA | V = 0,230 / separação 0,105 |
| Course / code_module | ambos | PERMITIDA | Contexto forte (OULAD: 11,5%–44,5%); usar com cuidado para não "aprender a turma" |
| Displaced, Daytime/evening | UCI | PERMITIDA | 0,129 / 0,055 |
| Escolaridade/ocupação dos pais | UCI | PERMITIDA | V ≈ 0,16–0,18 |
| gender, disability | ambos | PERMITIDA, **sensível** | Usar só após análise de equidade; associação existe (UCI: 0,245) mas pode reproduzir viés |

## Candidatas descartadas nesta lista (ver DATA_LEAKAGE_AUDIT.md)

`final_result`, `Target`, `date_unregistration`, qualquer agregado do período inteiro,
notas de prova final, resultados do 2º semestre (UCI), resultados do 1º semestre em
cortes do 1º semestre, indicador de `date_registration` ausente.

Em quarentena, aguardando decisão: `Tuition fees up to date`, `Debtor`,
`Scholarship holder`, variáveis macroeconômicas (UCI).
