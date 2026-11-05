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

## Situação atual — Fase 01 (inicialização e pesquisa de dados)

- ✅ Estrutura do repositório, `.gitignore`, ambiente virtual
- ✅ Pesquisa e avaliação de fontes brasileiras e internacionais
- ✅ Download e inspeção preliminar dos datasets UCI 697 e OULAD
- ⏸️ UFRN aguardando autorização de download
- ⛔ Nenhum modelo treinado; nenhuma API implementada (por definição desta fase)

## Tecnologias previstas

| Etapa | Tecnologia |
|---|---|
| Linguagem | Python 3.12 |
| Dados | pandas (atual); formato colunar a avaliar |
| Testes | pytest |
| Modelagem | a definir com justificativa na fase de modelagem |
| API | provavelmente FastAPI (a decidir) |
| Cliente | App EduGuard em Flutter (repositório separado) |

## Estrutura

```
eduguard-ia/
├── data/
│   ├── raw/            # originais imutáveis (não versionados) + manifestos *.source.json
│   ├── interim/        # intermediários
│   ├── processed/      # prontos para modelagem
│   └── external/       # dados de contexto (ex.: indicadores INEP)
├── docs/               # documentação do projeto
├── notebooks/          # exploração
├── research/
│   └── inspection/     # perfis exploratórios gerados por script
├── src/
│   ├── data/           # registry, download, inspeção
│   ├── preprocessing/
│   ├── feature_engineering/
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

## Documentação

- [docs/DATASET_RESEARCH.md](docs/DATASET_RESEARCH.md) — pesquisa e comparação de datasets
- [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) — dicionário preliminar
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) — proveniência, licenças e datas de acesso
- [docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md) — arquitetura planejada
- [docs/TECHNICAL_DECISIONS.md](docs/TECHNICAL_DECISIONS.md) — decisões e pendências

## Licença dos dados

Cada dataset mantém sua licença original (ver `docs/DATA_SOURCES.md`). Os dados
não são redistribuídos por este repositório.
