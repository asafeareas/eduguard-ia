# Registro de Fontes de Dados

Registro de proveniência de cada dataset: origem, licença, data de acesso e
instruções de obtenção. O registro programático equivalente está em
`src/data/registry.py`; cada download gera também um manifesto
`data/raw/<source_id>/<source_id>.source.json` (tamanho e SHA-256 de cada arquivo).

**Política:** arquivos originais ficam em `data/raw/` e nunca são editados. Nada
em `data/` é versionado no Git, exceto os manifestos `*.source.json`.

## Baixados

### uci_dropout_697 — Predict Students' Dropout and Academic Success

| Campo | Valor |
|---|---|
| URL de acesso | https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |
| URL de download | https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip |
| Licença | CC BY 4.0 — atribuição obrigatória |
| Citação | Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89 |
| Data de acesso | 2026-11-05T17:14:50Z |
| Arquivos | `predict_students_dropout_and_academic_success.zip` — 533.344 bytes — SHA-256 `e90e55fd65ec462ae283ebeb2cca409319e3460ed898d8754f62fb35cc83a65d` |
| | `extracted/data.csv` — 533.230 bytes — CSV (`;`, UTF-8 BOM) — SHA-256 `3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae` |
| Como obter | `python -m src.data.download uci_dropout_697` |

### oulad_349 — Open University Learning Analytics Dataset

Download autorizado pelo responsável do projeto em 2026-11-05.

| Campo | Valor |
|---|---|
| URL | https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset |
| Download | https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip (espelho OU: http://schools.stem.open.ac.uk/cdn/files/anonymisedData.zip — HTTP, não usado) |
| Licença | CC BY 4.0 |
| Citação | Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171. https://doi.org/10.1038/sdata.2017.171 |
| Data de acesso | 2026-11-05T17:19:46Z |
| Como obter | `python -m src.data.download oulad_349 --approved` |

| Arquivo | Bytes | SHA-256 |
|---|---|---|
| `open_university_learning_analytics_dataset.zip` | 46.748.244 | `f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4` |
| `extracted/studentVle.csv` | 453.836.331 | `52668253d876c5becbcb72185977152700cecab2942aca807fecc3dd54b937f0` |
| `extracted/studentAssessment.csv` | 5.690.483 | `b1839e17618a5de36b6d189a242d3123f7939a56c8c9ccef0c17b9199c614a6b` |
| `extracted/studentInfo.csv` | 3.462.763 | `815fbc3e2de29f79900bc63f0345a35db62c9146afaa1d86e7107b8b3beffc60` |
| `extracted/studentRegistration.csv` | 1.132.550 | `bbc87a0de1fe2a9ec6decce1ca6c1a02cf17b54ad3bdcc673fc6413b41c61d47` |
| `extracted/vle.csv` | 270.612 | `d7a146497edd0e47ef536144b58796c07b554002d3753e848b8e4cd8a56f4c38` |
| `extracted/OULAD.names` | 12.278 | `e497bc5232f385ece30a7e964b0339bd9b74ba9b9083508b66628fed9dc23ebe` |
| `extracted/assessments.csv` | 8.211 | `9ce92381e4a0ac457f8e251b2eb2c179ed51e023ab30d1d671a0268bd62316ba` |
| `extracted/courses.csv` | 526 | `4f16eee7454b15e109b0a21a0e43be820e6846ed6f9301bb7feb5ab5ad737a75` |

Total em disco: ~511 MB (ZIP + extraído).

## Aguardando autorização

### ufrn_open_data — UFRN Dados Abertos

| Campo | Valor |
|---|---|
| Portal | https://dados.ufrn.br/ (API CKAN `/api/3/action/package_show?id=...`) |
| Conjuntos | `situacoes-de-discentes` (~6,5 MB/semestre), `matriculas-componentes` (~65–77 MB/semestre), `turmas`, `dados-socio-economicos-de-discentes` |
| Licença | "Outra (Aberta)" — termos **não verificados** |
| Dados pessoais | IDs pseudonimizados (hash). **Excluir o conjunto `discentes`** (nomes e matrículas reais) |
| Autenticação | Não exige |
| Proposta | Baixar uma janela limitada (ex.: ingressos 2014–2019 de graduação presencial) após aprovação; script dedicado a ser escrito na próxima fase |

## Não baixados (baixa compatibilidade)

| Fonte | URL | Motivo |
|---|---|---|
| INEP — Censo da Educação Superior | https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-da-educacao-superior | Dados públicos agregados; sem rótulo individual; tamanho desconhecido |
| INEP — Indicadores de Fluxo | https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais/indicadores-de-fluxo-da-educacao-superior | Agregado curso × coorte; pode entrar depois como contexto |
| Plataforma Nilo Peçanha | https://www.gov.br/mec/pt-br/pnp | Granularidade não verificada; hosts de dados inacessíveis no teste |
| KDD Cup 2015 / XuetangX | https://www.biendata.xyz/competition/kddcup2015/data/ | Licença desconhecida; evasão de MOOC |
| xAPI-Edu-Data | Kaggle (exige conta) | Sem rótulo de evasão; licença desconhecida |
