# Definição dos Targets — EduGuard-IA

> Decisões aprovadas pelo responsável do projeto na Fase 03 (2026-11-05).
> Implementação: `src/feature_engineering/oulad_features.py` (`build_week`) e
> `src/feature_engineering/uci_admission.py`. Testes: `tests/test_phase03.py`.

## 1. OULAD — target principal: `target_withdrawn`

| final_result | target_withdrawn |
|---|---|
| Withdrawn | **1** |
| Pass | 0 |
| Distinction | 0 |
| Fail | 0 |

- **Significado:** o estudante, ainda matriculado no corte, **cancelará a matrícula no
  módulo** depois do corte.
- **O que NÃO significa:** evasão definitiva da universidade. A OULAD registra o
  cancelamento de um **módulo** (~9 meses) de ensino a distância britânico. Os resultados não
  devem ser apresentados como comprovação de previsão de evasão institucional.
- **Reprovação não é evasão** neste target. Limitação conhecida (Fase 02): 41,7% dos `Fail`
  pararam de acessar o AVA antes da metade do módulo, ou seja, parte deles abandonou sem
  cancelamento formal e está rotulada como 0.

## 2. OULAD — target secundário: `target_adverse_outcome`

| final_result | target_adverse_outcome |
|---|---|
| Withdrawn | **1** |
| Fail | **1** |
| Pass | 0 |
| Distinction | 0 |

- **Significado:** resultado acadêmico adverso no módulo (cancelar **ou** reprovar).
- **Não é** evasão institucional. Serve como experimento comparativo, mais próximo da ideia
  pedagógica de "estudante que precisa de apoio".
- Por construção, `target_withdrawn = 1` ⇒ `target_adverse_outcome = 1` (verificado em teste).

Os dois targets estão nas **mesmas linhas** de cada conjunto (S2, S4, S6). O "conjunto do
target secundário" é a mesma tabela usando a outra coluna de rótulo.

## 3. População elegível (vale para os dois targets)

Um estudante entra no conjunto do corte *t* somente se:

1. estava matriculado até *t* (`date_registration ≤ t`, ou data desconhecida);
2. **não** cancelou até *t* (`date_unregistration` ausente ou `> t`);
3. tem rótulo e data consistentes (excluídos: 93 `Withdrawn` sem data e 9 não-`Withdrawn` com data).

Consequência: o target mede **evasão futura**. Quem já saiu não é "previsto".
Limitação: os 26% de cancelamentos ocorridos **antes do dia 0** nunca entram em nenhum corte.

## 4. UCI 697 — target: `target_dropout`

| Target original | target_dropout |
|---|---|
| Dropout | **1** |
| Graduate | 0 |
| Enrolled | **excluído** do conjunto supervisionado |

- `Enrolled` = ainda matriculado ao fim da duração normal do curso: **desfecho censurado**.
  Tratar como 0 introduziria viés (alguns ainda podem abandonar). As 794 linhas ficam em
  `uci_enrolled_unlabelled.parquet`, sem rótulo, para análises de sensibilidade futuras.
- Limitação: excluir `Enrolled` remove o grupo intermediário e tende a deixar as métricas
  **otimistas** em relação a uma aplicação real, em que todos os ingressantes seriam avaliados.
- Momento da previsão: **ingresso**. O experimento é independente da OULAD.

## 5. Uso correto

- Targets são **apenas rótulos**: `split_roles(df, meta, target)` devolve X sem nenhuma
  coluna de target, ID ou auditoria (testado).
- `final_result` e `date_unregistration` existem nos arquivos **somente como auditoria** e
  nunca podem ser usados como features.
