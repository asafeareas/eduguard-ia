# Auditoria de Equidade — Fase 04

> Semana 4, target principal, teste 2014 sem recorrentes (14.185 registros). Thresholds de alerta
> definidos na partição de calibração (RL 0,234; GB 0,227). Código: `src/modeling/fairness.py`.
> Gênero e deficiência **não são preditores** (decisão Q16): foram usados apenas para
> segmentar os resultados, depois das previsões. Nunca entraram na seleção de modelos, nos
> hiperparâmetros, nos thresholds, na calibração ou na imputação.
> IC 95% por bootstrap de estudantes (300 reamostragens). Grupos com menos de 100 registros ou
> menos de 20 positivos são marcados como amostra insuficiente.
>
> **Diferenças observadas não comprovam discriminação.** Elas indicam onde investigar.

## 1. Regressão logística (modelo de referência)

| Grupo | Registros (estudantes) | Cancelaram | Observado | Previsto médio | Recall [IC] | Precision | FNR | FPR [IC] | Brier |
|---|---|---|---|---|---|---|---|---|---|
| **Gênero F** | 6.243 (6.038) | 1.032 | 16,5% | 17,7% | 0,415 [0,386–0,443] | 0,308 | 0,585 | 0,185 [0,174–0,195] | 0,131 |
| **Gênero M** | 7.942 (7.040) | 1.534 | 19,3% | 21,3% | 0,538 [0,517–0,562] | 0,299 | 0,462 | 0,302 [0,292–0,313] | 0,148 |
| **Deficiência N** | 12.884 (11.883) | 2.235 | 17,3% | 19,6% | 0,485 [0,462–0,503] | 0,291 | 0,515 | 0,248 [0,240–0,255] | 0,136 |
| **Deficiência Y** | 1.301 (1.195) | 331 | **25,4%** | **20,4%** | 0,514 [0,469–0,567] | 0,400 | 0,486 | 0,263 [0,237–0,288] | 0,178 |
| IMD 0–20% (mais privação) | 2.771 (2.566) | 542 | 19,6% | 21,9% | 0,541 [0,503–0,585] | 0,303 | 0,459 | 0,302 [0,283–0,322] | 0,150 |
| IMD 20–40% | 3.048 | 604 | 19,8% | 20,6% | 0,475 | 0,306 | 0,525 | 0,266 | 0,151 |
| IMD 40–60% | 2.798 | 513 | 18,3% | 19,8% | 0,497 | 0,304 | 0,503 | 0,255 | 0,141 |
| IMD 60–80% | 2.607 | 454 | 17,4% | 18,5% | 0,482 | 0,318 | 0,518 | 0,218 | 0,135 |
| IMD 80–100% (menos privação) | 2.412 (2.204) | 374 | 15,5% | 17,5% | 0,436 [0,387–0,489] | 0,287 | 0,564 | 0,199 [0,184–0,215] | 0,124 |
| IMD ausente | 549 | 79 | 14,4% | 18,9% | 0,456 | 0,234 | 0,544 | 0,251 | 0,122 |
| Idade 0–35 | 9.866 | 1.806 | 18,3% | 20,2% | 0,489 | 0,295 | 0,511 | 0,262 | 0,142 |
| Idade 35–55 | 4.216 | 742 | 17,6% | 18,7% | 0,487 | 0,321 | 0,513 | 0,219 | 0,136 |
| Idade 55+ ⚠️ | 103 (92) | 18 | 17,5% | 18,9% | 0,444 | 0,258 | 0,556 | 0,271 | 0,138 |

⚠️ Idade 55+: amostra insuficiente (18 cancelamentos). Não tirar conclusões.

![Equidade RL](../reports/modeling/figures/oulad_week04_fairness_logistic_regression.png)

## 2. Gradient boosting (resumo)

| Grupo | Recall | FPR | Previsto × observado |
|---|---|---|---|
| F / M | 0,342 / 0,470 | 0,170 / 0,263 | 15,8% × 16,5% / 19,0% × 19,3% |
| Deficiência N / Y | 0,415 / 0,444 | 0,220 / 0,232 | 17,6% × 17,3% / **17,8% × 25,4%** |
| IMD mais / menos privação | 0,502 / 0,329 | 0,304 / 0,170 | 19,5% × 19,6% / 16,3% × 15,5% |

![Equidade GB](../reports/modeling/figures/oulad_week04_fairness_gradient_boosting.png)

## 3. Achados

1. **Gênero:** o modelo sinaliza homens com mais frequência, com recall maior (0,54 × 0,42) e
   também mais falsos positivos (FPR 0,30 × 0,19); os ICs não se sobrepõem. A taxa observada
   também é maior entre homens (19,3% × 16,5%), mas a diferença de FPR vai além disso: mulheres
   que vão cancelar são menos detectadas, e homens que não vão cancelar recebem mais alertas.
   Gênero não é preditor, então a diferença vem de variáveis correlacionadas (módulo, região,
   padrão de engajamento).
2. **Deficiência — achado mais relevante:** estudantes com deficiência cancelam mais (25,4%), e o
   modelo **subestima** esse risco (previsto 20,4% na RL e 17,8% no GB). Recall e FPR são
   parecidos entre os grupos, mas a probabilidade exibida para esse grupo seria baixa demais
   (erro de calibração de cerca de −5 a −8 pontos percentuais). Excluir a variável (Q16) não
   elimina essa diferença. Ela deve ser tratada na próxima fase, por exemplo com auditoria de
   calibração por grupo antes de qualquer uso e sem incluir a variável como preditor.
3. **Privação socioeconômica (IMD):** quanto maior a privação, maior o recall e maior o FPR
   (0,30 × 0,20 entre os extremos). Mais alertas para estudantes de áreas mais pobres: parte disso
   corresponde a uma taxa observada maior (19,6% × 15,5%), parte é excesso de falsos positivos.
4. **Idade:** sem diferenças relevantes entre 0–35 e 35–55. O grupo 55+ é pequeno demais para concluir.

## 4. Com × sem `adm_imd_band` (S4)

| Modelo | PR-AUC com | PR-AUC sem | Diferença com − sem [IC 95%] |
|---|---|---|---|
| Regressão logística | 0,306 | 0,303 | +0,003 [+0,0004; +0,006] |
| Gradient boosting | 0,282 | 0,285 | −0,003 [−0,008; +0,002] |

Sem a variável, a regressão logística **iguala os falsos positivos** entre as faixas de IMD
(FPR 0,253 nos mais privados × 0,238 nos menos privados) e a calibração por faixa fica dentro de
±4 pontos percentuais (exceto IMD ausente: +7 p.p.). O ganho de PR-AUC de manter a variável é estatisticamente detectável, mas
**irrelevante na prática** (0,003).

**Recomendação (para decisão do responsável):** considerar retirar `adm_imd_band` do modelo
principal na próxima fase. O custo preditivo é desprezível, e com ela os alertas se concentram
em estudantes de áreas mais pobres, com excesso de falsos positivos, o que pode reproduzir
desigualdades socioeconômicas. A variável continuaria disponível para auditoria.

## 5. UCI (ingresso) — resumo

| Grupo (hold-out de 726) | n | Observado | Previsto | Recall | FPR |
|---|---|---|---|---|---|
| Gênero F | 438 | 29,7% | 35,4% | 0,73 | 0,34 |
| Gênero M | 288 | 53,5% | 47,5% | 0,77 | 0,44 |
| Idade ≤20 | 422 | 26,8% | 30,6% | 0,61 | 0,28 |
| Idade >25 | 174 | 66,7% | 62,2% | 0,88 | **0,81** |
| Necessidades educativas especiais Y ⚠️ | 9 | — | — | — | — |

Na UCI, a idade domina: **81%** dos estudantes com mais de 25 anos que se formaram receberiam um
alerta falso. Isso mostra o risco de modelos baseados só em ingresso, que rotulam grupos inteiros.
O grupo com necessidades especiais tem apenas 9 pessoas no teste: nenhuma conclusão é possível.

## 6. Limitações

- Uma única divisão de teste (2014); ICs refletem só a variação amostral daquele ano.
- Equidade avaliada para um threshold específico. Outro threshold mudaria recall e FPR por grupo.
- Os grupos usam as categorias disponíveis na OULAD (gênero binário, deficiência autodeclarada,
  IMD da área de residência), que são aproximações imperfeitas.
- Nenhuma técnica de mitigação foi aplicada nesta fase. As diferenças estão documentadas, não corrigidas.

---

# Atualização da Fase 05 (modelo v2, sem `adm_imd_band`)

> Semana 4, target principal, teste 2014 sem recorrentes. Modelo **v2**: gênero, deficiência e
> IMD fora dos preditores. Thresholds da partição de calibração (RL 0,211; GB 0,194). IC 95% por
> bootstrap de estudantes (1.000 reamostragens no diagnóstico de deficiência). Os números da v1
> acima ficam preservados para rastreabilidade. Resultados: `reports/phase05/`.

## A. Consolidação das disparidades

| Dimensão | v1 | v2 | Situação |
|---|---|---|---|
| IMD: amplitude de FPR entre faixas | 0,103 (0,199–0,302) | **0,015** (0,296–0,311; ICs sobrepostos) | **Resolvida** pela remoção (Q17) |
| IMD: calibração por faixa | +0,007 a +0,023 | +0,003 a +0,041 (ausente +0,072 [+0,040; +0,103]) | Leve superestimação nas faixas menos privadas e na ausente |
| Gênero: FPR M × F (RL) | 0,302 × 0,185 | **0,369 [0,359–0,381] × 0,232 [0,220–0,244]** | **Persiste** |
| Gênero: recall M × F (RL) | 0,538 × 0,415 | 0,606 [0,584–0,631] × 0,472 [0,441–0,502] | Persiste |
| Gênero: calibração | — | M +0,022 [+0,013; +0,031] · F +0,018 [+0,008; +0,027] | Semelhante entre os grupos |
| Deficiência: calibração (Y) | −0,051 | **−0,049 [−0,072; −0,027]** | **Persiste** (ver B) |

Com gênero, a calibração é parecida entre os grupos, mas o modelo distribui mais alertas (e
mais falsos) para homens. Isso decorre de variáveis correlacionadas com gênero, como módulo e
padrão de uso do AVA. Mitigar sem usar gênero exigiria rever essas features ou o desenho do
alerta, e isso fica para decisão.

## B. Estudantes com deficiência (Q18)

### Tamanho e representatividade

| Partição | Registros | Com deficiência | % | Taxa de cancelamento: deficiência × demais |
|---|---|---|---|---|
| Treino 2013 | 7.085 | 697 | 9,8% | 25,8% × 16,5% |
| Validação 2013 | 2.362 | 224 | 9,5% | 23,7% × 14,7% |
| Calibração 2013 | 2.360 | 202 | 8,6% | 27,2% × 16,5% |
| **Teste 2014** | 14.185 | **1.301** | 9,2% | **25,4% × 17,3%** |
| Recorrentes 2014 (secundário) | 1.449 | 173 | 11,9% | 35,3% × 30,5% |

O grupo **está representado no treino** (cerca de 10%, o mesmo do teste) e a taxa maior é
**consistente** em todas as partições. Não é um artefato de amostra. Com 331 casos positivos no
teste, a menor diferença de recall detectável (α = 0,05; poder de 80%) é de **cerca de 8 pontos
percentuais**. Diferenças menores não podem ser afirmadas nem descartadas.

### Calibração (subestimação do risco)

| Corte | Modelo | Observado (Y) | Previsto (Y) | Gap [IC 95%] | Gap dos demais |
|---|---|---|---|---|---|
| S2 | RL | 26,9% | 19,3% | **−0,077 [−0,100; −0,053]** | −0,001 |
| S4 | RL | 25,4% | 20,5% | **−0,049 [−0,072; −0,027]** | +0,027 |
| S4 | GB | 25,4% | 17,6% | **−0,078 [−0,100; −0,057]** | +0,003 |
| S6 | RL | 23,5% | 14,6% | **−0,089 [−0,112; −0,067]** | −0,023 |

A subestimação aparece em **todos os cortes e nos dois modelos**, e os ICs excluem zero.

![Calibração por deficiência](../reports/phase05/figures/v2_disability_calibration.png)

Por quintil (RL, S4), o problema se concentra nos riscos intermediários e altos: previsto 0,18 → observado 0,28; previsto 0,37 → observado 0,47.

### Recall, falsos negativos e falsos positivos (S4)

| Modelo | Grupo | Recall [IC 95%] | FNR | FPR | Precision | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|
| RL | N | 0,550 [0,528–0,570] | 0,450 | 0,307 | 0,273 | 0,293 | 0,137 |
| RL | **Y** | 0,565 [0,511–0,614] | 0,435 | 0,320 | 0,376 | 0,399 | 0,179 |
| GB | N | 0,538 [0,518–0,558] | 0,462 | 0,320 | 0,261 | 0,273 | 0,138 |
| GB | **Y** | 0,535 [0,481–0,587] | 0,465 | 0,311 | 0,370 | 0,390 | 0,184 |

Fonte: `reports/phase05/disability_audit.json`.

**Interpretação:** no threshold único, a **taxa de detecção é semelhante** (recall e FNR sem
diferença detectável), porque a taxa de base maior compensa a subestimação. A probabilidade
exibida, porém, fica **cerca de 5 pontos abaixo** da realidade. Em faixas de risco, estudantes
com deficiência seriam classificados abaixo do risco real, e com threshold de capacidade (top *k*%) seriam priorizados de menos.

### O mix de módulos explica?

Pouco. Padronizando pela distribuição de módulos do teste, o gap vai de −0,049 para **−0,041**.
A subestimação se concentra nos módulos **DDD (−0,105, n = 249)**, **FFF (−0,094, n = 285)** e
GGG (−0,050). Em BBB e CCC fica perto de zero. A causa provável é que a deficiência
acrescenta risco além do que o engajamento e as entregas capturam. Isso é coerente com
excluir a variável (Q16), mas mostra que **excluir não garante equidade**.

### Ações (sem thresholds por grupo)

1. **Acompanhamento humano independente do score** (`docs/GOVERNANCE_AND_PRIVACY.md` §3): o núcleo
   de acessibilidade segue seus próprios critérios, sem depender do alerta.
2. **Monitorar a calibração por grupo** a cada período e reportar à equipe.
3. **Investigar na próxima fase**, se autorizado: features de contexto que expliquem o risco
   adicional (ex.: carga de créditos, quarentenada na Fase 03) e recalibração geral. Nenhuma
   dessas opções usa deficiência como preditor.
4. **Não apresentar** a probabilidade desse grupo como "baixo risco" sem revisão humana.

## C. Limitações estatísticas

- Idade 55+ (103 registros, 18 cancelamentos) e grupos pequenos da UCI: amostra insuficiente.
- Os testes de diferença por grupo têm poder limitado (diferença mínima detectável de cerca de 8 p.p. para deficiência).
- Um único ano de teste. Os ICs não capturam mudanças entre anos.
