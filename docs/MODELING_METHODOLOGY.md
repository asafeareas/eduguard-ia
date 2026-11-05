# Metodologia de Modelagem — Fase 04

> 2026-11-05 · Branch `feat/model-training` (base: `feat/feature-engineering`).
> Código: `src/modeling/`. Execução completa: `python -m src.modeling.run_phase04`.
> Experimentos científicos: **nenhuma previsão operacional foi gerada**.

## 1. Experimentos

| Experimento | Dados | Target | Momento | Análise |
|---|---|---|---|---|
| OULAD S2 | `oulad_week_02` | `target_withdrawn` (cancelamento de módulo) e, separadamente, `target_adverse_outcome` | dia 13 | comparação, calibração, interpretação |
| **OULAD S4** | `oulad_week_04` | idem | dia 27 | **completa** (+ faixas de risco, equidade, balanceamento, sem IMD, recorrentes) |
| OULAD S6 | `oulad_week_06` | idem | dia 41 | comparação, calibração, interpretação |
| UCI ingresso | `uci_admission` | `target_dropout` | ingresso | independente |

São **13 experimentos independentes de treino**: 6 OULAD (3 cortes × 2 targets) + 1 UCI, além das
variantes de S4. Cada experimento tem divisão, ajuste, seleção, calibração e teste próprios.
Cortes e targets nunca são misturados num mesmo treino; OULAD e UCI nunca são combinados.

## 2. Validação OULAD (decisão Q15)

```
2013 ──► estudantes divididos (GroupShuffleSplit, seed 42):
           treino 60%  ── ajuste dos modelos e baselines
           validação 20% ── seleção de hiperparâmetros, de modelo e do N da regra de inatividade
           calibração 20% ── escolha/ajuste do calibrador, threshold de alerta e faixas de risco
2014 ──► teste principal = 2014 SEM os estudantes que aparecem em 2013 (intocado até o fim)
         cenário secundário = 2014 SÓ com esses estudantes ("previamente observados")
```

- O `student_id` serve **apenas** para montar a divisão; não é feature.
- Uma terceira partição (calibração) foi criada dentro de 2013 para que calibrador, threshold e
  faixas **não** sejam ajustados nos mesmos dados usados para escolher o modelo, e nunca no teste.
- O modelo final de cada experimento é o ajustado no treino (60% de 2013). Custo: menos dados de
  treino; ganho: validação e calibração realmente independentes.
- Verificado em teste: nenhum estudante é compartilhado entre treino, validação, calibração e teste.

### Auditoria da divisão (registros; estudantes entre parênteses)

| | S2 | S4 | S6 |
|---|---|---|---|
| 2013 | 12.076 (11.947) | 11.807 (11.686) | 11.616 (11.502) |
| 2014 | 15.900 (14.632) | 15.634 (14.446) | 15.312 (14.201) |
| Estudantes em 2013 **e** 2014 | 1.421 | **1.368** | 1.274 |
| Registros excluídos do teste principal | 1.508 | 1.449 | 1.343 |
| Treino | 7.246 (7.167) | 7.085 (7.011) | 6.961 (6.900) |
| Validação | 2.414 (2.390) | 2.362 (2.337) | 2.329 (2.301) |
| Calibração | 2.416 (2.390) | 2.360 (2.338) | 2.326 (2.301) |
| **Teste principal** | **14.392 (13.211)** | **14.185 (13.078)** | **13.969 (12.927)** |
| Cenário secundário (recorrentes) | 1.508 (1.421) | 1.449 (1.368) | 1.343 (1.274) |

O número de recorrentes varia por corte porque a população elegível muda (1.368 em S4, como na Fase 03).

### Distribuição das classes — `target_withdrawn` (taxa positiva)

| | S2 | S4 | S6 |
|---|---|---|---|
| Treino | 19,0% | 17,4% | 15,6% |
| Validação | 18,1% | 15,5% | 16,0% |
| Calibração | 19,4% | 17,4% | 15,4% |
| 2014 antes da exclusão | 20,8% | 19,3% | 17,6% |
| **Teste principal** | **19,4%** | **18,1%** | **16,7%** |
| Recorrentes (secundário) | 33,6% | 31,1% | 27,3% |

`target_adverse_outcome` no teste principal: 42,8% (S2), 41,9% (S4), 40,9% (S6); entre recorrentes: 62,0%, 60,5%, 58,2%.

**A exclusão dos recorrentes altera a distribuição?** Pouco no total (−1,2 a −1,4 ponto percentual),
mas de forma sistemática: os recorrentes (provavelmente quem repete o módulo) têm quase o **dobro**
da taxa de cancelamento. O teste principal, portanto, sub-representa um grupo de alto risco, que
é avaliado à parte no cenário secundário (S4).

## 3. Validação UCI

A UCI não tem ano de ingresso confiável, então **nenhuma validação cronológica é simulada**:

- Hold-out estratificado de 20% (726 estudantes), intocado até o fim.
- Nos 80% restantes: validação cruzada estratificada de 5 folds para escolher modelo e
  hiperparâmetros. As previsões fora do fold do modelo escolhido definem calibrador e threshold.
  O modelo final é reajustado nos 80%.
- Verificação de robustez **não temporal**: deixar uma coorte (`cohort_proxy`) de fora.
- Sem estudantes repetidos (uma linha por estudante): a divisão por linha equivale à divisão por estudante.

## 4. Prevenção de vazamento

| Risco | Controle | Teste |
|---|---|---|
| Eventos após o corte | Conjuntos da Fase 03 (invariância já testada) + verificação do hash de conteúdo contra o manifesto antes de treinar | `test_real_results_consistent_with_manifest_and_cutoff` |
| IDs, targets, auditoria como features | `predictor_columns` usa só `feature_columns` dos metadados | `test_predictors_exclude_ids_targets_audit_and_sensitive` |
| Gênero e deficiência (Q16) | Removidos dos preditores (`EXCLUDED_SENSITIVE`), usados só na auditoria | idem + `test_experiment_outputs_respect_q16` |
| Transformações aprendidas fora do treino | Tudo dentro de `Pipeline`/`ColumnTransformer`, ajustado no treino | `test_imputation_fitted_on_train_only` |
| Uso do teste para seleção, calibração ou threshold | Seleção na validação, calibração e threshold na partição de calibração | `test_test_labels_never_influence_learning` (embaralhar rótulos do teste não muda nada aprendido) |
| Indicador de ausência que revela o desfecho | Sem indicador para `adm_registration_lead_days` (quarentena da Fase 03) | `test_no_missing_indicator_for_quarantined_registration_missingness` |
| Mesmo estudante em treino e teste | Divisão por estudante + exclusão de recorrentes | `test_temporal_split_and_recurrent_exclusion`, `test_internal_split_is_by_student` |

## 5. Pré-processamento (dentro do pipeline)

| Tipo de coluna | Regressão logística | Gradient boosting |
|---|---|---|
| Numérica, contagem, ordinal, binária | mediana do treino + padronização | mediana do treino |
| `asm_submission_rate`, `asm_mean_days_before_deadline`, `adm_imd_band` | + indicador de ausência | + indicador de ausência |
| Nominal (`ctx_code_module`, `ctx_presentation_start`, `adm_region`; códigos da UCI) | categoria "missing" + one-hot com `handle_unknown="ignore"` | idem |

- Indicadores de ausência só onde a ausência tem significado estrutural: nenhuma avaliação vencida,
  nenhuma entrega ou IMD ausente na fonte. A ausência **não** é tratada como "zero comportamento".
- Nenhum balanceamento no protocolo principal (distribuição natural). `class_weight="balanced"`
  foi testado só como variante comparativa em S4, ajustado no treino.

## 6. Modelos e seleção

| Modelo | Grade (selecionado por PR-AUC na validação) |
|---|---|
| Prevalência | taxa positiva do treino |
| Taxa do módulo | taxa por módulo no treino (mín. 30 registros; senão taxa global) |
| Regra de inatividade | alto risco se `eng_days_since_last_activity ≥ N`; N de 1 a corte+2, escolhido por F1 na validação |
| Regressão logística | L1/L2 × C ∈ {0,01; 0,03; 0,1; 0,3; 1; 3}, solver liblinear |
| Gradient boosting (`HistGradientBoostingClassifier`) | learning_rate {0,05; 0,1} × folhas {15, 31} × min_samples_leaf {50, 200} × iterações {150, 300}; L2 = 1; sem early stopping |

**Por que gradient boosting (e não Random Forest):** captura não linearidades e interações
(ex.: inatividade × módulo); lida nativamente com escalas diferentes; tende a gerar
probabilidades melhor calibradas que Random Forest; treina em segundos com ~7 mil linhas; a
regularização (folhas, `min_samples_leaf`, L2) controla a complexidade. Early stopping foi
desligado porque usaria uma divisão aleatória por linha (não por estudante) dentro do treino.

## 7. Calibração, threshold e faixas

- Calibrador ∈ {nenhum, Platt, isotônico}, escolhido pelo Brier fora do fold (5 folds por
  estudante) **dentro da partição de calibração**. Só troca o "nenhum" se o ganho for maior que
  0,5% do Brier.
- Threshold de alerta = o que maximiza F1 nas probabilidades calibradas da partição de calibração.
- Faixas de risco: ver `docs/RISK_CLASSIFICATION.md`.

## 8. Comparações estatísticas

Bootstrap pareado por **estudante** (1.000 reamostragens) no teste: diferença de PR-AUC entre
gradient boosting e regressão logística, e entre a regressão logística e o melhor baseline na
validação. Um modelo só é declarado melhor se o IC 95% da diferença excluir zero **e** a
diferença for relevante na prática.

## 9. Reprodutibilidade

Seed 42 em todos os sorteios. Cada `results.json` registra: dataset, hash do conteúdo (comparado
com o manifesto), target, corte, preditores, seed, auditoria da divisão, grades e parâmetros
escolhidos, métricas, commit e versões das bibliotecas. Modelos (`models.joblib`) e previsões do
teste (`test_predictions.parquet`) não são versionados. Teste de determinismo:
`test_experiment_is_deterministic`.


---

## 10. Fase 05 — verificação da divisão por estudante

> Questão levantada na revisão: na semana 4, treino + validação + calibração = 7.085 + 2.362 +
> 2.360 = 11.807, ou seja, o número de **registros** e não o de estudantes (11.686). A divisão é
> mesmo por estudante?

**Verificação no código** (`src/modeling/splitting.py`): `split_by_student` usa
`GroupShuffleSplit(groups=df["student_id"])`. As frações (20% para calibração; depois 25% do
restante para validação) são de **estudantes**, e todas as linhas de um estudante vão para a
mesma parte. `check_disjoint` interrompe a execução se houver estudante em duas partes.

**Verificação nos dados reais** (`reports/phase05/split_integrity.json`):

| | S2 | S4 | S6 |
|---|---|---|---|
| Registros 2013 = soma dos registros das 3 partes | 12.076 = 12.076 | 11.807 = 11.807 | 11.616 = 11.616 |
| Estudantes 2013 = soma dos estudantes das 3 partes | 11.947 = 11.947 | **11.686 = 7.011 + 2.337 + 2.338** | 11.502 = 11.502 |
| Estudantes com mais de 1 registro em 2013 (dois módulos) | 129 | **121** | 114 |
| Estudantes em mais de uma parte | 0 | 0 | 0 |
| Estudantes compartilhados com o teste | 0 | 0 | 0 |

**Conclusão:** não há inconsistência nem vazamento. As tabelas da Fase 04 mostram os
**registros** fora dos parênteses e os **estudantes** dentro deles. As duas somas fecham. A
diferença de 121 corresponde a estudantes matriculados em dois módulos em 2013, sempre com os
dois registros na mesma parte. Nenhuma correção ou reexecução foi necessária. A verificação
virou teste com dados reais (`test_real_split_is_by_student_and_counts_reconcile`).

## 11. Fase 05 — versões dos preditores

| Versão | Exclui dos preditores (fica só para auditoria) | S2 / S4 / S6 | Resultados |
|---|---|---|---|
| phase04-v1 | gênero, deficiência | 31 / 47 / 49 | `reports/modeling/` (inalterados) |
| **phase05-v2** (atual) | gênero, deficiência, `adm_imd_band` | **30 / 46 / 48** | `reports/modeling_v2/` |

Uso: `run_oulad_experiment(..., model_version="phase05-v2")`. O protocolo é o mesmo nas duas.
Ver `docs/MODEL_V2_EVALUATION.md`.
