# Arquitetura Planejada — EduGuard-IA

> Documento de planejamento. **Nenhuma API foi implementada nesta fase.**

## 1. Visão geral

```
 ┌────────────────────┐     ┌───────────────────────────────────────────┐
 │ Fontes de dados    │     │ EduGuard-IA (Python)                      │
 │ (UFRN, OULAD, UCI, │──►  │                                           │
 │  dados da IES)     │     │  src/data            ingestão + manifesto │
 └────────────────────┘     │  src/preprocessing   limpeza, tipos       │
                            │  src/feature_eng.    features por CORTE   │
                            │                      TEMPORAL (semana k)  │
                            │  src/models          treino (fase futura) │
                            │  src/evaluation      validação temporal   │
                            │  src/prediction      inferência + fatores │
                            └───────────────┬───────────────────────────┘
                                            │ modelo versionado
                                            ▼
                            ┌───────────────────────────────────────────┐
                            │ API REST (fase futura)                    │
                            │  POST /v1/predictions                     │
                            │  GET  /v1/students/{pseudo_id}/risk       │
                            │  GET  /v1/model/info                      │
                            └───────────────┬───────────────────────────┘
                                            │ HTTPS + JSON (token)
                                            ▼
                            ┌───────────────────────────────────────────┐
                            │ App EduGuard (Flutter) — repositório      │
                            │ separado, não alterado nesta fase         │
                            └───────────────────────────────────────────┘
```

## 2. Princípio central: features "as-of"

Toda feature é calculada em função de um **instante de corte** (`cutoff`), por
exemplo "dia 28 do semestre". A regra que o código das próximas fases deve
obedecer:

> Uma feature para o corte *t* só pode usar eventos com timestamp ≤ *t*.
> Estudantes cujo desfecho (ex.: cancelamento) ocorreu antes de *t* são
> removidos da população daquele corte.

Assim, o mesmo pipeline gera conjuntos de treino para semanas 2, 4, 6… e permite
medir **quanto a antecedência custa em desempenho**.

## 3. Módulos

| Módulo | Responsabilidade | Estado |
|---|---|---|
| `src/data/registry.py` | Catálogo de fontes (URL, licença, política de download) | **implementado** |
| `src/data/download.py` | Download, extração segura, manifesto com SHA-256 | **implementado** |
| `src/data/inspect_dataset.py` | Perfil exploratório somente leitura | **implementado** |
| `src/data/loaders.py` | Leitura padronizada (missing `"?"`, BOM, dtypes compactos do AVA) | **Fase 02** |
| `src/data/quality.py` | Unicidade de chaves, órfãos, consistência rótulo × datas | **Fase 02** |
| `src/feature_engineering/temporal.py` | População e agregados *as-of* por corte (semana *k* → dia 7k−1) | **Fase 02** (exploratório; seleção final na Fase 03) |
| `src/analysis/eda.py` | Estatísticas descritivas (taxas, Cramér's V, AUC univariada, trajetórias) | **Fase 02** |
| `src/analysis/leakage.py` | Catálogo de vazamento (PERMITIDA/CONDICIONAL/PROIBIDA) + regras empíricas | **Fase 02** |
| `src/analysis/plots.py` | Gráficos (matplotlib/seaborn) | **Fase 02** |
| `src/analysis/run_phase02.py` / `build_notebooks.py` | Pipeline reprodutível e notebooks | **Fase 02** |
| `src/feature_engineering/catalog.py` | Catálogo de features (origem, disponibilidade, cortes, tipo, domínio, quarentena) e papéis das colunas | **Fase 03** |
| `src/feature_engineering/oulad_features.py` | Conjuntos OULAD por corte (população, engajamento, avaliações sem notas, contexto, ingresso, targets) | **Fase 03** |
| `src/feature_engineering/uci_admission.py` | Conjunto UCI de ingresso (Enrolled separado) | **Fase 03** |
| `src/feature_engineering/build_datasets.py` | Gera Parquet + metadados + manifestos; `split_roles` separa X/y sem IDs/auditoria | **Fase 03** |
| `src/preprocessing/` | — | não usado: o pré-processamento ficou em `src/modeling/preprocessing.py`, junto dos pipelines |
| `src/modeling/` | Divisão (Q15), pipelines, baselines, treino, avaliação, calibração, faixas, equidade, interpretação, UCI | **Fase 04** |
| `src/modeling/preprocessing.py::MODEL_VERSIONS` | Versões dos preditores: phase04-v1 e **phase05-v2** (sem IMD; atual) | **Fase 05** |
| `src/modeling/thresholds.py` | Threshold por capacidade, por custo e benefício líquido (metodologia; nada operacional) | **Fase 05** |
| `src/validation/` | Correspondência OULAD→Brasil, contrato de dados (LGPD), compatibilidade, avaliação externa sem retreino, PSI | **Fase 05** |
| `src/models/` | Treino e persistência | vazio (sem algoritmo escolhido) |
| `src/evaluation/` | Validação por coorte/tempo, calibração, métricas por subgrupo | vazio |
| `src/prediction/` | Probabilidade → faixa de risco → fatores explicativos | vazio |

## 4. Contrato de saída previsto (para o app Flutter)

Esboço — os nomes podem mudar quando a API for desenhada.

```json
{
  "student_pseudo_id": "a51a7575…",
  "cutoff": {"term": "2026.1", "week": 4},
  "dropout_probability": 0.71,
  "risk_level": "high",
  "top_factors": [
    {"feature": "absences_rate_week4", "direction": "increases_risk", "value": 0.32},
    {"feature": "units_without_evaluation", "direction": "increases_risk", "value": 2}
  ],
  "model_version": "x.y.z",
  "generated_at": "2026-03-30T12:00:00Z"
}
```

- `risk_level`: limiares de probabilidade (baixo/moderado/alto) a serem definidos
  com a equipe pedagógica, considerando capacidade de atendimento — **pendente**.
- `top_factors`: explicação local (método a definir) — linguagem orientada à ação,
  nunca determinística.

## 5. Integração prevista com Flutter

- Framework provável: FastAPI (tipagem, OpenAPI automático gera cliente Dart).
  **Decisão adiada** para a fase da API.
- Comunicação HTTPS/JSON, autenticação por token emitido pelo backend institucional.
- O app recebe somente o ID pseudonimizado e os campos acima; dados pessoais ficam
  no sistema acadêmico da instituição.
- Previsões em lote (job semanal) + endpoint de consulta, evitando inferência
  pesada sob demanda no celular.

## 6. Requisitos não funcionais

- **LGPD:** minimização de dados, pseudonimização, registro de finalidade.
- **Reprodutibilidade:** manifesto com SHA-256, versões fixadas, seeds.
- **Auditabilidade:** cada previsão registra a versão do modelo e o corte usado.
- **Equidade:** métricas por subgrupo (sexo, idade, modalidade de ingresso) antes
  de qualquer uso real.

## 7. Governança (preservada e reforçada na Fase 05)

- O EduGuard-IA continua **independente** do app Flutter. A comunicação futura será por **API
  REST**, que **ainda não foi autorizada**.
- O modelo **não toma decisões acadêmicas**: entrega estimativas de risco com incerteza para
  revisão humana. A equipe pedagógica define os critérios de acompanhamento
  (`docs/THRESHOLD_METHODOLOGY.md`).
- Dados brasileiros só com autorização, conforme o contrato de dados e a LGPD
  (`docs/GOVERNANCE_AND_PRIVACY.md`). Sensíveis (gênero, deficiência, socioeconômico) ficam só para auditoria.
- O modelo que eventualmente for servido precisa ter validação na instituição de destino. O
  modelo OULAD não é transferível diretamente (`docs/BRAZILIAN_FEATURE_MAPPING.md`).
