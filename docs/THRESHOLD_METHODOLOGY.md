# Metodologia para Definição Futura de Thresholds (Q19)

> Fase 05 · 2026-11-05. **Os thresholds atuais são apenas referências experimentais.** Nenhum
> threshold definitivo será definido sem a equipe pedagógica. Probabilidades **não** viram
> rótulos de estudantes.
> Ferramentas: `src/modeling/thresholds.py`. Ilustração com o modelo v2 da semana 4 (RL), na
> **partição de calibração de 2013** (2.360 registros, 410 cancelamentos; o teste de 2014 não
> é usado): `reports/phase05/threshold_methodology.json`.

## 1. Três perguntas para a equipe pedagógica

| Pergunta | Parâmetro | Ferramenta |
|---|---|---|
| Quantos estudantes a equipe consegue acompanhar por período? | capacidade *k* (% da turma) | `capacity_threshold(p, k)` |
| Quanto pior é **deixar de alertar** um estudante que vai cancelar do que alertar quem não vai? | razão de custos FN : FP | `cost_optimal_threshold(c_fn, c_fp)` = c_fp / (c_fp + c_fn), válido para probabilidades calibradas |
| Quanto uma intervenção realmente evita cancelamentos? | efetividade *e* (0–1, medida em piloto) | `consequences(y, p, t, effectiveness=e)` |

O threshold escolhido deve ser o **menor** compatível com a capacidade. A razão de custos e
a efetividade servem para discutir se vale a pena ampliar a capacidade.

## 2. Ilustração — por capacidade de atendimento

| Capacidade | Threshold | Alertas | Alertas corretos | Falsos alertas | Cancelamentos não alertados | Precision | Recall | Alertas por caso real |
|---|---|---|---|---|---|---|---|---|
| 5% | 0,405 | 118 | 62 | 56 | 348 | 0,53 | 0,15 | 1,9 |
| 10% | 0,303 | 236 | 109 | 127 | 301 | 0,46 | 0,27 | 2,2 |
| 20% | 0,233 | 472 | 169 | 303 | 241 | 0,36 | 0,41 | 2,8 |
| 30% | 0,205 | 708 | 216 | 492 | 194 | 0,31 | 0,53 | 3,3 |

## 3. Ilustração — por razão de custos (FN : FP)

| FN custa | Threshold | % sinalizados | Precision | Recall |
|---|---|---|---|---|
| 2× FP | 0,333 | 8,7% | 0,47 | 0,23 |
| 3× FP | 0,250 | 15,8% | 0,40 | 0,36 |
| 5× FP | 0,167 | 50,6% | 0,24 | 0,70 |
| 10× FP | 0,091 | 92,3% | 0,19 | 0,98 |

Se a equipe considera que perder um caso custa muito mais que um contato desnecessário, o
threshold ótimo sinaliza quase todos, ou seja, o modelo deixa de ajudar a priorizar. Esse é um
resultado direto do sinal modesto do modelo.

## 4. Benefício líquido (curva de decisão)

Benefício líquido = VP/n − FP/n × t/(1 − t). O modelo supera a estratégia "alertar todos" para
thresholds a partir de 0,10, e o benefício se aproxima de zero acima de 0,45. **Só há utilidade
na faixa aproximada de 0,10 a 0,40.**

| t | 0,10 | 0,15 | 0,20 | 0,25 | 0,30 | 0,35 | 0,40 | 0,45 |
|---|---|---|---|---|---|---|---|---|
| Modelo | 0,089 | 0,051 | 0,039 | 0,031 | 0,023 | 0,017 | 0,010 | 0,004 |
| Alertar todos | 0,082 | 0,028 | −0,033 | −0,102 | −0,180 | −0,271 | −0,377 | −0,502 |

## 5. Procedimento proposto (com a equipe)

1. **Definir a capacidade** realista por período e o tipo de intervenção (contato, tutoria, apoio).
2. **Calcular** o threshold de capacidade em dados de calibração **da própria instituição**
   (depois da validação brasileira; nunca com o teste).
3. **Mostrar à equipe** as consequências (tabelas acima) e as diferenças por grupo
   (`docs/FAIRNESS_AUDIT.md`) antes de aprovar.
4. **Piloto com revisão humana** de todos os alertas, registrando as intervenções e os resultados para estimar a efetividade.
5. **Revisar** a cada período: recalibrar, recalcular o threshold e auditar por grupo.
6. **Comunicação:** mostrar "risco estimado" com faixa e incerteza; nunca "vai evadir".

## 6. Restrições

- Sem thresholds por grupo (decisão atual).
- Sem ações automáticas a partir do score.
- Os números acima são da OULAD (cancelamento de módulo) e **não** valem para instituições brasileiras.
