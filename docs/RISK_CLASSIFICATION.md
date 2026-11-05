# Classificação de Risco — Proposta Preliminar

> Fase 04 · Semana 4, target principal (cancelamento de módulo). Código:
> `src/modeling/calibration.py` (`propose_risk_bands`, `assign_bands`).
>
> ⚠️ **Proposta experimental. Não deve ser usada para rotular estudantes reais nem para
> decisões institucionais automatizadas.**

> **Fase 05 (Q19):** estes limites são **referências experimentais**. Com o modelo v2 (sem IMD)
> os limites são os mesmos (baixo < 0,174; alto ≥ 0,348), e no teste ficam: baixo 47% (11,0%
> observado), moderado 44% (21,2%), alto 9% (39,6%). Thresholds definitivos dependem da
> metodologia de `docs/THRESHOLD_METHODOLOGY.md` e da equipe pedagógica.

## 1. Critério

As faixas são definidas sobre **probabilidades calibradas**, usando **apenas** dados de 2013
(treino para a prevalência e partição de calibração para o threshold). O teste de 2014 não foi
consultado; ele foi usado só para *verificar* as faixas depois.

| Faixa | Regra | Justificativa |
|---|---|---|
| **Baixo** | p < prevalência do treino | Abaixo do risco médio da população de referência |
| **Moderado** | prevalência ≤ p < limite alto | Acima da média, mas sem evidência forte |
| **Alto** | p ≥ máx(threshold de F1 na calibração; 2 × prevalência) | O threshold de F1 é o ponto de melhor equilíbrio de alerta; o piso de 2× garante que "alto" signifique pelo menos o dobro do risco médio |

Não foram usados quantis fixos (ex.: "os 10% do topo são alto risco"), porque eles criam
três categorias por construção, independentemente de haver risco real. Também não foi usada a
distribuição das próprias probabilidades como justificativa de calibração.

## 2. Limites obtidos (S4)

| Modelo | Baixo < | Alto ≥ | Origem do limite alto |
|---|---|---|---|
| Regressão logística | 0,174 | 0,348 | 2 × prevalência (o threshold de F1 = 0,234 ficou abaixo do piso) |
| Gradient boosting | 0,174 | 0,348 | idem (threshold de F1 = 0,227) |

## 3. Verificação

### Partição de calibração (2013, usada para definir as faixas)

| Faixa | RL: % estudantes / taxa observada | GB: % / taxa |
|---|---|---|
| Baixo | 59,1% / 11,1% | 60,6% / 10,7% |
| Moderado | 33,2% / 20,9% | 30,9% / 23,5% |
| Alto | 7,8% / **49,7%** | 8,5% / 42,8% |

### Teste 2014 (verificação, não usado para definir nada)

| Faixa | RL: registros (%) / taxa observada | GB: registros (%) / taxa |
|---|---|---|
| Baixo | 7.150 (50,4%) / **11,2%** | 8.057 (56,8%) / 12,1% |
| Moderado | 5.556 (39,2%) / **21,7%** | 5.204 (36,7%) / 24,5% |
| Alto | 1.479 (10,4%) / **38,1%** | 924 (6,5%) / 34,5% |

- A **ordenação se mantém** no teste: alto > moderado > baixo, com cerca de 3,4× de diferença entre alto e baixo.
- A faixa alta **perdeu precisão** de 2013 para 2014 (49,7% → 38,1% na RL): sinal de mudança
  entre anos e motivo para recalibrar periodicamente.
- Na faixa alta, **6 em cada 10 estudantes não cancelam**. "Alto risco" não significa
  "vai cancelar".
- Metade dos estudantes fica em "baixo" e ainda assim responde por cerca de 31% dos
  cancelamentos (7.150 × 11,2% ≈ 800 de 2.566).

## 4. Limitações para uso institucional

1. O target é cancelamento de **módulo** na Open University (EAD, Reino Unido), não evasão de uma
   instituição brasileira.
2. As faixas dependem da prevalência de referência (17,4%). Em outra instituição, a prevalência e
   os limites seriam outros.
3. Há diferenças de calibração por grupo (`docs/FAIRNESS_AUDIT.md`): estudantes com deficiência
   têm risco subestimado, e estudantes de áreas mais pobres recebem mais alertas falsos.
4. A capacidade de atendimento da equipe pedagógica não foi considerada. Na prática, o limite
   "alto" deveria ser ajustado ao número de estudantes que a equipe consegue acompanhar, junto
   com a equipe.
5. Faixas são apoio à conversa pedagógica, nunca rótulos permanentes. Qualquer uso futuro exige
   validação local, transparência para o estudante e revisão humana de cada caso.
