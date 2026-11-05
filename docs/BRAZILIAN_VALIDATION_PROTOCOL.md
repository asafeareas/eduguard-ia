# Protocolo de Validação Brasileira (Q20)

> Fase 05 · 2026-11-05 · Status: **protocolo pronto; validação empírica PENDENTE**. Não há
> autorização institucional para acessar dados brasileiros. Nenhum dado brasileiro foi
> acessado, baixado ou usado.
> Código de apoio: `src/validation/` (correspondência, contrato de dados, verificação de
> compatibilidade, avaliação externa sem retreino, deslocamento de distribuição).

## 0. Pré-condições (todas obrigatórias)

1. Autorização formal da instituição (ex.: UFRN), com finalidade descrita: pesquisa e
   validação científica, sem uso operacional.
2. Base legal da LGPD definida pelo controlador (ver `docs/GOVERNANCE_AND_PRIVACY.md`) e
   aprovação do comitê de ética quando aplicável.
3. Entrega conforme o **contrato de dados** (`src/validation/data_contract.py`): IDs
   pseudonimizados, nada de nome, CPF, e-mail ou matrícula real; dados sensíveis só para auditoria.
4. Ambiente controlado: acesso restrito, sem cópia para repositórios, sem upload a serviços externos.

## 1. Etapa A — recebimento e conformidade

| Passo | Ferramenta | Critério de aceite |
|---|---|---|
| Validar o arquivo | `data_contract.validate(df)` | `valid = True`; nenhum campo proibido; cancelamentos com data |
| Registrar proveniência | manifesto com SHA-256 (mesmo padrão de `src/data/download.py`) | hash registrado antes de qualquer análise |
| Inventário de campos | `data_contract.validate(...)["optional_present"]` | lista de campos opcionais efetivamente entregues |

## 2. Etapa B — decisão de transferência

```python
from src.validation.external import compatibility_report
rep = compatibility_report(model_predictors, available_fields)
```

- **Se `direct_transfer_possible = True`:** segue para a Etapa C (validação externa do modelo original).
- **Se `False`** (resultado esperado com os dados conhecidos; ver `docs/BRAZILIAN_FEATURE_MAPPING.md`):
  **a validação externa do modelo original é impossível** e isso deve ser declarado. Segue para a Etapa D.

Com o modelo v2 (S4, 46 preditores), só `adm_age_band` é diretamente transferível. A decisão
esperada hoje é, portanto, **não transferir**.

## 3. Etapa C — validação externa sem retreino (só se compatível)

1. Recalcular as features com **as mesmas regras** da Fase 03 (corte = fim da semana *k* do
   período letivo; só eventos com data ≤ corte; população = matriculados e não cancelados até o corte).
2. Aplicar o pipeline salvo **sem reajustar nada**: modelo, imputação, codificação, calibrador e threshold.
   `external.evaluate_external(bundle, model, df, target, group_cols=[...], reference=train_df, feature_types=...)`.
   A função recusa dados incompatíveis.
3. Métricas: PR-AUC, ROC-AUC, recall, precision, Brier, ECE, tabela de calibração, top 10% e 20%,
   com IC por bootstrap de estudantes.
4. Subgrupos: curso, período letivo, turno/modalidade, instituição/campus e os atributos de auditoria
   (sexo, deficiência, indicador socioeconômico), usando os critérios de amostra mínima da Fase 04
   (≥ 100 registros e ≥ 20 positivos).
5. Deslocamento de distribuição: PSI por feature numérica (> 0,25 = grande) e proporção de categorias
   nunca vistas nas nominais.
6. Entre instituições: repetir por instituição e comparar ICs. Nunca agregar instituições em uma só população.

## 4. Etapa D — adaptação ou novo treinamento (se incompatível)

Isso **não é** validação externa do modelo OULAD. É o desenvolvimento de um **novo modelo
brasileiro**, que reaproveita a **metodologia**:

| Elemento | Reaproveitado da OULAD | Redefinido para o Brasil |
|---|---|---|
| Rótulo | Estrutura binária, população por corte | "Cancelamento/trancamento de disciplina" ou "evasão do curso" (definições distintas; escolher com a equipe) |
| Cortes | Semanas 2/4/6, `cutoff = 7k − 1` | Contados do início das aulas do período letivo |
| Features | Família de indicadores (engajamento, entrega, recência, tendência) | Frequência/faltas datadas, notas por unidade com data, entregas no AVA, histórico de períodos anteriores |
| Validação | Temporal (anos anteriores → posterior), por estudante, partição de calibração | Ex.: treino 2016–2018 → teste 2019 (2020–2021 com análise separada por causa da pandemia) |
| Equidade | Auditoria por grupo com IC | + forma de ingresso/cotas como **auditoria** |
| Leakage | Testes de invariância temporal | Mesmos testes, adaptados ao novo esquema |

Uma comparação justa entre o "modelo brasileiro" e o "modelo OULAD" só é possível em nível de
**metodologia e ordem de grandeza**, não de métricas diretamente comparáveis.

## 5. Problemas previstos de generalização

1. **Unidade e rótulo diferentes:** módulo de 9 meses em EAD × disciplina semestral presencial × curso.
2. **AVA:** uso do AVA no presencial brasileiro pode ser opcional, e a ausência de cliques não
   significa desengajamento.
3. **Calendário:** semestres de cerca de 100 dias contra 234 a 269 dias. Semana 4 representa
   proporções diferentes do período.
4. **Frequência:** existe no Brasil e não existe na OULAD. Provavelmente é o sinal mais forte,
   e o modelo OULAD não sabe usá-lo.
5. **Notas sem data:** nos dados abertos da UFRN, notas e faltas não têm data de lançamento, o
   que impede cortes semanais rigorosos.
6. **Pandemia (2020–2021):** períodos atípicos devem ser analisados à parte.
7. **IDs:** é preciso confirmar se o pseudônimo é o mesmo entre as tabelas (pendência Q10).

## 6. Resultado desta fase

- Protocolo, contrato de dados e ferramentas: **prontos e testados** (`tests/test_phase05.py`).
- **Validação empírica brasileira: NÃO realizada.** Nenhuma métrica brasileira existe.
- Conclusão técnica antecipada: com os campos conhecidos, a **transferência direta do modelo
  OULAD é inviável**. O caminho recomendado é a Etapa D, sob a governança descrita.
