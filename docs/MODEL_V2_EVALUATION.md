# Modelo v2 — Remoção de `adm_imd_band` (Q17)

> Fase 05 · 2026-11-05. Versões dos preditores (`src/modeling/preprocessing.py::MODEL_VERSIONS`):
> **phase04-v1** = features − {gênero, deficiência}; **phase05-v2** = v1 − {`adm_imd_band`}
> (**modelo principal atual**). A faixa de privação continua nos datasets **só para auditoria**.
>
> Os resultados da Fase 04 (`reports/modeling/`) **não foram alterados**. A v2 está em
> `reports/modeling_v2/` e a comparação em `reports/phase05/v1_vs_v2_*`.
> Protocolo idêntico ao da Fase 04 (mesma divisão, mesma grade, seleção na validação,
> calibração e threshold na partição de calibração, teste 2014 intocado).
> Ainda é **predição de cancelamento de módulo**, não de evasão institucional.

## 1. Preditores por versão (reconciliação de contagens)

| Corte | Features no dataset (Fase 03) | v1 (Fase 04) | **v2 (Fase 05)** |
|---|---|---|---|
| S2 | 33 | 31 | **30** |
| S4 | 49 | 47 | **46** |
| S6 | 51 | 49 | **48** |

## 2. Métricas no teste 2014 (sem recorrentes) — target principal

| Corte | Modelo | PR-AUC v1 | **PR-AUC v2** [IC 95%] | Δ v2 − v1 [IC 95%, bootstrap pareado] | ROC-AUC v1 → v2 | Brier v1 → v2 | ECE v1 → v2 |
|---|---|---|---|---|---|---|---|
| S2 | RL | 0,271 | **0,267** [0,255–0,281] | −0,004 [−0,006; −0,002] | 0,629 → 0,624 | 0,152 → 0,152 | 0,014 → 0,017 |
| S2 | GB | 0,253 | 0,250 [0,239–0,264] | −0,004 [−0,007; −0,001] | 0,605 → 0,599 | 0,154 → 0,154 | 0,020 → 0,021 |
| **S4** | **RL** | 0,306 | **0,304** [0,289–0,321] | **−0,002 [−0,006; +0,001]** | 0,668 → 0,665 | 0,140 → 0,140 | 0,022 → 0,020 |
| S4 | GB | 0,282 | 0,285 [0,270–0,301] | +0,003 [−0,002; +0,008] | 0,651 → 0,651 | 0,142 → 0,142 | 0,015 → 0,013 |
| S6 | RL | 0,267 | **0,263** [0,249–0,276] | −0,005 [−0,009; −0,001] | 0,667 → 0,663 | 0,133 → 0,133 | 0,028 → 0,029 |
| S6 | GB | 0,271 | 0,266 [0,254–0,283] | −0,005 [−0,009; +0,000] | 0,657 → 0,654 | 0,134 → 0,133 | 0,030 → 0,015 |

**Leitura:** retirar a faixa de privação custa de **0 a 0,005** de PR-AUC. No corte principal
(S4) a diferença não é estatisticamente distinguível de zero, em nenhum dos dois modelos.
A calibração fica praticamente igual.

Target secundário (resultado adverso): perdas de 0,003 a 0,011 de PR-AUC (S4 RL: +0,003, empate).
Tabela completa: `reports/phase05/v1_vs_v2_metrics.csv`.

## 3. Threshold e matriz de confusão (S4, RL v2)

O threshold de alerta é re-selecionado na partição de calibração (2013): **0,211** na v2
contra 0,234 na v1. Por isso, recall e precision no threshold não são comparáveis um a um
entre as versões. Para comparar, use PR-AUC, top-k e calibração.

| | v1 | v2 |
|---|---|---|
| Threshold | 0,234 | 0,211 |
| VP / FP / FN / VN | 1.253 / 2.899 / 1.313 / 8.720 | 1.416 / 3.575 / 1.150 / 8.044 |
| Precision / recall | 0,302 / 0,488 | 0,284 / 0,552 |
| Top 10%: cancelamentos / recall / precision | 550 / 21,4% / 38,8% | 554 / 21,6% / 39,0% |
| Top 20% | 957 / 37,3% / 33,7% | 948 / 36,9% / 33,4% |
| Faixas no teste (baixo / moderado / alto): % e taxa observada | 50% (11,2%) / 39% (21,7%) / 10% (38,1%) | 47% (11,0%) / 44% (21,2%) / 9% (39,6%) |

## 4. Efeito nos subgrupos (S4, RL; teste 2014)

| Grupo | FPR v1 | FPR v2 | Recall v1 | Recall v2 | Calibração (prevista − observada) v1 → v2 |
|---|---|---|---|---|---|
| IMD 0–20% (mais privação) | 0,302 | 0,311 | 0,541 | 0,552 | +0,023 → +0,009 |
| IMD 80–100% (menos privação) | 0,199 | 0,299 | 0,436 | 0,545 | +0,020 → +0,041 |
| **Amplitude de FPR entre faixas de IMD (sem "ausente")** | **0,103** | **0,015** | — | — | — |
| Gênero F / M | 0,185 / 0,302 | 0,232 / 0,369 | 0,415 / 0,538 | 0,472 / 0,606 | sem mudança relevante |
| Deficiência N / Y | 0,248 / 0,263 | 0,307 / 0,320 | 0,485 / 0,514 | 0,550 / 0,565 | Y: −0,051 → **−0,049** |

- **IMD:** a disparidade de falsos positivos entre as faixas de privação **desaparece**
  (amplitude de 0,10 para 0,015; os ICs se sobrepõem, ver `docs/FAIRNESS_AUDIT.md`).
  A calibração por faixa fica dentro de +0,04, exceto IMD ausente (+0,07).
- **Gênero:** a disparidade persiste (FPR M 0,37 [0,36–0,38] × F 0,23 [0,22–0,24]). Ela não
  dependia do IMD.
- **Deficiência:** a subestimação de risco persiste (ver Q18 em `docs/FAIRNESS_AUDIT.md`).
- A queda de FPR geral entre v1 e v2 é consequência do threshold mais baixo, não da remoção do IMD.

## 5. Interpretação (S4, v2)

A ordem dos grupos de features se mantém: avaliações (queda de PR-AUC de 0,067) >
engajamento (0,022) > contexto (0,013) > ingresso (0,012). Os maiores coeficientes da RL
continuam sendo o calendário e a identidade do módulo, a 1ª avaliação vencida e a
escolaridade prévia.

## 6. Decisão registrada

A v2 é o **modelo principal** (TD-42). O custo preditivo é desprezível e a disparidade de
falsos positivos por privação é eliminada. A v1 continua disponível e rastreável para comparação.

## 7. Rastreabilidade

- Commits dos resultados v2: `e3ded28` (S2 principal, S4 principal) e `30da34f` (demais). O código de modelagem
  é **idêntico** entre os dois (`git diff e3ded28 30da34f -- src/modeling src/feature_engineering src/data`
  vazio). O alerta `src_uncommitted_changes = true` no S4 principal se deve a arquivos novos de
  `src/validation/` criados durante a execução, que não são usados pelo treinamento.
- Hashes de conteúdo dos datasets iguais aos manifestos (verificado em teste).
