# EduGuard-IA

Módulo de **Academic Analytics** do EduGuard: transforma dados acadêmicos em
**alertas antecipados de risco de evasão** estudantil. Projeto Python independente,
que futuramente disponibilizará previsões ao aplicativo mobile EduGuard (Flutter)
por meio de uma API REST.

## Objetivo

Classificação supervisionada (`EVADIU = 1`, `NÃO EVADIU = 0`) que, **nas primeiras
semanas do semestre**, entregue:

1. probabilidade estimada de evasão;
2. classificação de risco (baixo, moderado, alto);
3. principais fatores associados ao risco;
4. informações de apoio à equipe pedagógica.

O modelo só pode aprender com informações que **existiriam no momento da previsão**
— controle de vazamento temporal é requisito, não detalhe.

## Situação atual

**Fase 01 — inicialização e pesquisa de dados** (na `main`)
- ✅ Estrutura do repositório, `.gitignore`, ambiente virtual
- ✅ Pesquisa e avaliação de fontes brasileiras e internacionais
- ✅ Download e inspeção preliminar dos datasets UCI 697 e OULAD
- ⏸️ UFRN aguardando autorização de download

**Fase 02 — exploração de dados e viabilidade temporal** (branch `feat/data-exploration`)
- ✅ EDA de OULAD e UCI 697, com 17 gráficos e 3 notebooks executados
- ✅ Agregação *as-of* por corte temporal (semanas 2/4/6), testada contra vazamento
- ✅ Auditoria de vazamento e lista preliminar de features candidatas

**Fase 03 — engenharia de features e conjuntos de modelagem** (branch `feat/feature-engineering`)
- ✅ Conjuntos OULAD por corte: semana 2 (33 features), 4 (49) e 6 (51), com target principal
  (cancelamento do módulo) e secundário (resultado adverso)
- ✅ Conjunto UCI de ingresso (18 features; Enrolled excluído)
- ✅ Manifestos com hashes, testes de invariância temporal e de vazamento
- ⛔ Nenhum modelo treinado; nenhuma API implementada (por definição das fases)

## Tecnologias previstas

| Etapa | Tecnologia |
|---|---|
| Linguagem | Python 3.12 |
| Dados | pandas (atual); formato colunar a avaliar |
| Visualização / notebooks | matplotlib, seaborn, Jupyter (nbformat/nbconvert) |
| Armazenamento processado | Parquet (pyarrow) |
| Testes | pytest |
| Modelagem | a definir com justificativa na fase de modelagem |
| API | provavelmente FastAPI (a decidir) |
| Cliente | App EduGuard em Flutter (repositório separado) |

## Estrutura

```
eduguard-ia/
├── data/
│   ├── raw/            # originais imutáveis (não versionados) + manifestos *.source.json
│   ├── interim/        # intermediários (não versionados): cliques semanais, snapshots por corte
│   ├── processed/      # conjuntos de modelagem (Parquet não versionado; manifestos versionados)
│   └── external/       # dados de contexto (ex.: indicadores INEP)
├── docs/               # documentação do projeto
├── notebooks/          # 01 OULAD EDA · 02 UCI EDA · 03 viabilidade temporal e vazamento
├── research/
│   ├── figures/        # gráficos da Fase 02
│   └── inspection/     # perfis e resultados agregados (JSON/CSV)
├── src/
│   ├── analysis/       # EDA, auditoria de vazamento, gráficos, pipeline da Fase 02
│   ├── data/           # registry, download, loaders, qualidade, inspeção
│   ├── preprocessing/
│   ├── feature_engineering/  # catálogo de features, construção dos conjuntos por corte (OULAD) e ingresso (UCI)
│   ├── models/
│   ├── evaluation/
│   └── prediction/
├── tests/
├── requirements.txt
└── README.md
```

## Fontes de dados

Resumo — detalhes em [docs/DATASET_RESEARCH.md](docs/DATASET_RESEARCH.md) e
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

| Fonte | Compatibilidade | Status |
|---|---|---|
| OULAD — Open University (CC BY 4.0) | Alta | **baixado** |
| UFRN Dados Abertos (situações + matrículas em componentes) | Alta (condicionada) | aguardando autorização |
| UCI 697 — Predict Students' Dropout (CC BY 4.0) | Parcial | **baixado** |
| INEP — Censo da Educação Superior / Indicadores de Fluxo | Baixa (agregado) | não baixado |

## Configuração do ambiente

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Obter os dados e verificar:

```bash
python -m src.data.download uci_dropout_697
python -m src.data.inspect_dataset data/raw/uci_dropout_697/extracted/data.csv --sep ";" --target Target --out research/inspection/uci_dropout_697_profile.json
python -m src.data.download oulad_349 --approved   # ~47 MB de download, ~511 MB em disco
python -m src.data.inspect_oulad
python -m pytest -q
```

Fase 02 (análise exploratória, ~1 min):

```bash
python -m src.analysis.run_phase02        # JSON/CSV em research/inspection, PNG em research/figures
python -m src.analysis.build_notebooks    # gera e executa os notebooks
```

Fase 03 (conjuntos de modelagem, ~20 s):

```bash
python -m src.feature_engineering.build_datasets   # data/processed/** + manifestos
```

## Documentação

- [docs/DATASET_RESEARCH.md](docs/DATASET_RESEARCH.md) — pesquisa e comparação de datasets
- [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) — dicionário preliminar
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) — proveniência, licenças e datas de acesso
- [docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md) — arquitetura planejada
- [docs/TECHNICAL_DECISIONS.md](docs/TECHNICAL_DECISIONS.md) — decisões e pendências
- [docs/EXPLORATORY_DATA_ANALYSIS.md](docs/EXPLORATORY_DATA_ANALYSIS.md) — análise exploratória (Fase 02)
- [docs/TEMPORAL_FEASIBILITY.md](docs/TEMPORAL_FEASIBILITY.md) — viabilidade nas semanas 2, 4 e 6
- [docs/DATA_LEAKAGE_AUDIT.md](docs/DATA_LEAKAGE_AUDIT.md) — auditoria de vazamento
- [docs/DATASET_COMPARISON.md](docs/DATASET_COMPARISON.md) — OULAD × UCI
- [docs/FEATURE_CANDIDATES.md](docs/FEATURE_CANDIDATES.md) — features candidatas (preliminar)
- [docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md) — catálogo de features (Fase 03)
- [docs/TARGET_DEFINITION.md](docs/TARGET_DEFINITION.md) — targets aprovados
- [docs/MODELING_DATASETS.md](docs/MODELING_DATASETS.md) — conjuntos de modelagem
- [docs/FEATURE_VALIDATION.md](docs/FEATURE_VALIDATION.md) — testes temporais, de integridade e de vazamento

## Licença dos dados

Cada dataset mantém sua licença original (ver `docs/DATA_SOURCES.md`). Os dados
não são redistribuídos por este repositório.
