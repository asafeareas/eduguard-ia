# Conjuntos de Dados para Modelagem — Fase 03

> Gerados em 2026-11-05 por `python -m src.feature_engineering.build_datasets` (~20 s).
> Os arquivos Parquet **não são versionados**. Metadados e manifestos (`*.metadata.json`,
> `*.manifest.json`) são versionados e contêm: fonte, versão da fonte, hash SHA-256 dos
> arquivos brutos (verificado a cada execução), data de processamento, corte, número de
> registros e features, definição dos targets, distribuição das classes, regras de população,
> hash do arquivo de saída, hash do conteúdo e commit do código.

## 1. Estrutura

```
data/processed/
├── oulad/
│   ├── week_02/oulad_week_02.parquet     (1,3 MB)
│   ├── week_04/oulad_week_04.parquet     (1,9 MB)
│   ├── week_06/oulad_week_06.parquet     (2,1 MB)
│   └── metadata/oulad_week_0k.{metadata,manifest}.json
└── uci/admission/
    ├── uci_admission.parquet              (supervisionado)
    ├── uci_enrolled_unlabelled.parquet    (Enrolled, sem rótulo)
    └── uci_admission.{metadata,manifest}.json
```

Colunas de cada conjunto OULAD = IDs (4) + features + targets (2) + auditoria (8).

## 2. OULAD

| | Semana 2 | Semana 4 (principal) | Semana 6 |
|---|---|---|---|
| Corte (último dia) | 13 | 27 | 41 |
| Registros (estudante × módulo) | **27.976** | **27.441** | **26.928** |
| Estudantes distintos | 25.158 | 24.764 | 24.429 |
| Features | **33** | **49** | **51** |
| `target_withdrawn` = 1 | 5.590 (**20,0%**) | 5.026 (**18,3%**) | 4.511 (**16,8%**) |
| `target_adverse_outcome` = 1 | 12.615 (45,1%) | 12.062 (44,0%) | 11.548 (42,9%) |
| Pass / Distinction / Fail / Withdrawn | 12.339 / 3.022 / 7.025 / 5.590 | 12.356 / 3.023 / 7.036 / 5.026 | 12.356 / 3.024 / 7.037 / 4.511 |

### População (de 32.593 matrículas)

| Exclusão | S2 | S4 | S6 |
|---|---|---|---|
| Rótulo/data inconsistentes | 102 | 102 | 102 |
| Ainda não matriculados no corte | 58 | 18 | 12 |
| Já cancelados até o corte | 4.457 | 5.032 | 5.551 |
| **Elegíveis** | **27.976** | **27.441** | **26.928** |

### Por ano da apresentação (para validação temporal)

| Ano | S2: n / % Withdrawn | S4: n / % Withdrawn | S6: n / % Withdrawn |
|---|---|---|---|
| 2013 | 12.076 / 18,9% | 11.807 / 17,0% | 11.616 / 15,7% |
| 2014 | 15.900 / 20,8% | 15.634 / 19,3% | 15.312 / 17,6% |

Taxa de Withdrawn em S4 por módulo: GGG 7,9% · AAA 13,4% · EEE 13,2% · BBB 14,1% ·
FFF 18,5% · DDD 23,7% · CCC 29,6%.

**Atenção para a próxima fase:** 1.368 estudantes aparecem em 2013 **e** em 2014 (S4), e
1.274 têm mais de uma matrícula no mesmo ano. A divisão treino/teste precisa ser **temporal
e por estudante**.

### Valores ausentes (mantidos, sem imputação)

| Feature | S2 | S4 | S6 | Motivo |
|---|---|---|---|---|
| `adm_imd_band` | 1.036 | 1.027 | 1.012 | Ausente na fonte |
| `adm_registration_lead_days` | 7 | 7 | 7 | Data de matrícula ausente na fonte |
| `asm_submission_rate` | — | 5.445 | 2.873 | Indefinida sem avaliação vencida |
| `asm_mean_days_before_deadline` | — | 7.674 | 4.644 | Indefinida sem nenhuma entrega |

Qualidade (`research/inspection/phase03_quality_*.json`): 0 IDs duplicados, 0 features
constantes, 0 valores fora do domínio, targets sem ausentes e só com {0, 1}.

## 3. UCI 697 — ingresso

| | Valor |
|---|---|
| Registros supervisionados | **3.630** (Dropout + Graduate) |
| Enrolled separados (sem rótulo) | 794 |
| Features | **18** (só ingresso) |
| `target_dropout` = 1 | 1.421 (**39,1%**) |
| `target_dropout` = 0 | 2.209 (60,9%) |
| Coortes (`cohort_proxy`) | 10, de 291 a 452 estudantes |
| Ausentes / duplicatas / fora do domínio | 0 / 0 / 0 |

## 4. Como carregar (sem vazamento)

```python
from src.feature_engineering.build_datasets import load_dataset, split_roles
df, meta = load_dataset("data/processed/oulad/week_04/oulad_week_04.parquet",
                        "data/processed/oulad/metadata/oulad_week_04.metadata.json")
X, y = split_roles(df, meta, "target_withdrawn")   # X contém apenas features
groups = df[meta["grouping_columns_for_validation"]["student"]]
```

## 5. Limitações

- O target da OULAD é cancelamento de **módulo**, não evasão da universidade.
- Notas estão fora dos conjuntos (sem data de divulgação). O sinal acadêmico vem só de
  entregas, atrasos e não entregas.
- Linhas duplicadas do AVA (7,4%) foram mantidas (soma de cliques), conforme a Fase 02.
- UCI: sem nenhuma informação comportamental; o grupo `Enrolled` foi excluído.
- Nenhum dos conjuntos representa o contexto presencial brasileiro.
