# Validação das Features — Fase 03

> Suíte completa: `python -m pytest -q` → **85 testes passando** (Fases 01–03).
> Testes desta fase: `tests/test_phase03.py` e `tests/test_phase03_docs.py`.

## 1. Testes temporais

| Teste | O que comprova |
|---|---|
| `test_temporal_invariance_to_post_cutoff_events[2/4/6]` | Para cada corte, adicionar cliques posteriores ao corte, entregas posteriores ao corte e mudar a data futura de cancelamento e o resultado final de um estudante elegível **não altera nenhuma feature** |
| `test_real_data_invariance_on_one_presentation` | O mesmo, em dados reais (AAA 2013J): 2.000 cliques sintéticos com 777 cliques após o dia 27 não alteram as features da semana 4 |
| `test_audit_bounds` / `test_processed_dataset_matches_manifest_and_rules[2/4/6]` | `max_vle_day_used` e `max_submission_day_used` ≤ corte; nenhum estudante cancelado até o corte; nenhum matriculado depois do corte (sintético e nos Parquets reais) |
| `test_week_feature_sets_respect_availability` | S2 sem nenhuma feature de avaliação; semanais só até a semana do corte |
| `test_recent_window_features` | Janelas de 14 dias e variação semanal calculadas nos dias certos |

## 2. Testes de leakage

| Teste | O que comprova |
|---|---|
| `test_features_do_not_depend_on_outcome` | Trocar `final_result` (Pass → Distinction, Distinction → Fail) não altera nenhuma feature |
| `test_split_roles_excludes_ids_targets_and_audit` | X nunca contém IDs, targets ou auditoria; pedir `final_result` como target falha |
| `test_catalog_is_consistent` | Nenhuma feature tem nome de ID, target ou auditoria; features em quarentena não têm corte |
| `test_week_feature_sets_respect_availability` | Notas, `studied_credits` e indicador de matrícula ausente não aparecem em nenhum conjunto |
| `test_uci_admission_dataset` / `test_uci_guard_rejects_forbidden_features` | UCI sem variáveis de semestre, financeiras ou macro; o construtor recusa listas com variáveis proibidas |

## 3. Testes de cálculo, população e targets

| Teste | O que comprova |
|---|---|
| `test_engagement_feature_values` | Valores calculados à mão: total, pré-início, dias ativos, médias, semanais, inclinação, desvio, CV, recência, entropia |
| `test_never_active_student_gets_deterministic_values` | Estudante sem cliques: recência = corte + 1, `eng_any_activity = 0` |
| `test_assessment_feature_values` | Vencidas, entregues, no prazo, atrasadas, falta, peso perdido, 1ª avaliação, *banked* fora do denominador |
| `test_submission_rate_undefined_without_due` | Taxa fica ausente (não 0) quando nada venceu |
| `test_static_feature_mapping` | Codificações ordinais (IMD "10-20", escolaridade, idade) |
| `test_eligibility_rules` | Exclusões: já cancelado, ainda não matriculado, cancelado **no** dia do corte, rótulos inconsistentes; soma fecha com a base |
| `test_target_definitions` | Withdrawn → (1,1), Fail → (0,1), Pass/Distinction → (0,0) |

## 4. Integridade e reprodutibilidade

| Teste | O que comprova |
|---|---|
| `test_build_is_deterministic_and_order_independent` | Duas execuções e entradas embaralhadas geram tabelas idênticas e o mesmo hash de conteúdo |
| `test_processed_week2_is_reproducible_from_raw` | Reconstruir S2 a partir dos dados brutos reproduz exatamente o Parquet gravado (inclusive tipos) |
| `test_processed_dataset_matches_manifest_and_rules` / `test_processed_uci_matches_manifest` | SHA-256 do arquivo e hash do conteúdo batem com o manifesto; colunas = catálogo; IDs únicos; targets em {0,1} |
| `test_raw_files_unchanged_since_download` (Fase 02) | Os dados brutos continuam idênticos ao download |
| `build_datasets.verified_raw_files` | Interrompe a construção se o hash de algum arquivo bruto mudar |

## 5. Problemas encontrados e corrigidos durante a fase

1. **Tipo dependente da entrada:** `eng_total_clicks` saía `int32` ou `int64` conforme o tipo
   da entrada (detectado pelo teste de invariância com dados reais). Corrigido com
   `canonical_dtypes`: o tipo vem do catálogo.
2. **Hash de conteúdo antes da gravação:** o hash calculado em memória diferia do Parquet
   relido. Agora é calculado sobre os dados **relidos do disco**.
3. **Identidade algébrica em S2:** `eng_last_week_change` = `eng_weekly_trend_slope` quando
   k = 2 → feature restrita a S4/S6.
4. **Resultados *banked*:** contá-los como "não entregues" penalizaria quem repete o módulo.
   Como estão todos no dia −1, saem do denominador daquele estudante.
5. **Erro no valor esperado de um teste** (entropia: soma manual de cliques errada). O teste
   foi corrigido após verificar o cálculo à mão; o código não mudou.

## 6. Limitações

- Testes provam que **não usamos o futuro dentro da OULAD**. Eles não provam que os
  registros da fonte foram gravados na data informada.
- "Avaliação disponibilizada" é aproximada pelo prazo, porque a data de abertura não existe.
- `eng_clicks_pct_in_presentation` depende de quem está elegível no mesmo corte e na mesma
  apresentação (sem rótulos). Em produção, a população deve ser a da turma corrente.
- Estudantes repetidos entre 2013 e 2014 (1.368) exigem divisão por estudante na Fase 04.
