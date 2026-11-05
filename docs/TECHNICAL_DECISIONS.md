# Decisões Técnicas — EduGuard-IA

Formato: cada decisão tem contexto, decisão e justificativa. Questões em aberto
ficam na seção final.

## Decisões tomadas (Fase 01)

### TD-01 — Projeto independente do app Flutter
- **Decisão:** repositório próprio em `C:\Users\asafe\eduguard-ia`, Git inicializado
  na branch `main`. O repositório `eduguard-dev-mobile` **não foi tocado**.
- **Justificativa:** ciclos de vida diferentes; integração futura via API REST.

### TD-02 — Estrutura de diretórios
- Seguida a estrutura sugerida, com acréscimo de `data/interim/` (dados
  intermediários entre bruto e processado) e `research/inspection/` (perfis
  exploratórios gerados por script).

### TD-03 — Python 3.12 + ambiente virtual `.venv`
- Python 3.12.10 já instalado no sistema. Ambiente isolado em `.venv` (ignorado no Git).

### TD-04 — Dependências mínimas
- Apenas `pandas` e `pytest`. Download usa a biblioteca padrão (`urllib`), sem
  `requests`. Bibliotecas de ML entram somente quando o algoritmo for escolhido
  com justificativa.

### TD-05 — Dados fora do Git
- `data/**` ignorado; versionados apenas `.gitkeep` e manifestos `*.source.json`.
  Também ignorados: `.env`, `kaggle.json`, chaves, modelos serializados.

### TD-06 — Originais imutáveis + manifesto de proveniência
- O ZIP original é preservado; a extração vai para `extracted/`. Cada arquivo
  recebe tamanho e SHA-256. A extração rejeita caminhos que escapem do diretório
  (proteção contra *zip slip*).

### TD-07 — Download automático apenas para fontes pequenas e liberadas
- `auto_download=True` somente para UCI 697 (520 KB, CC BY 4.0, sem autenticação).
- OULAD e UFRN exigem `--approved` explícito (volume e/ou dados pseudonimizados).

### TD-08 — Conjunto `discentes` da UFRN excluído
- A pré-visualização mostrou **nome completo e número de matrícula reais**. A
  amostra foi apagada imediatamente e o conjunto não será utilizado.

### TD-09 — Não fundir datasets
- UCI (Portugal, presencial, curso completo), OULAD (Reino Unido, EAD, módulo) e
  UFRN (Brasil, presencial) têm populações e definições de evasão diferentes.
  Serão experimentos separados, comparados por protocolo comum.

### TD-10 — UCI 697 restrita a previsão "início do 2º semestre"
- Variáveis do 1º/2º semestre são de **fim de semestre**; usá-las para prever no
  início do mesmo semestre seria vazamento. `Debtor`, `Tuition fees up to date` e
  `Scholarship holder` ficam em quarentena por não terem data de coleta documentada.

### TD-11 — Código em inglês, documentação em português
- Conforme requisito do projeto.

### TD-12 — OULAD baixada (autorizada em 2026-11-05)
- Inspeção em `src/data/inspect_oulad.py`: lê `studentVle.csv` em blocos de
  2 milhões de linhas (não carrega 454 MB de uma vez).
- Missing codificado como `"?"` → leitura com `na_values="?"`.

### TD-13 — População por corte temporal (OULAD)
- Para um corte no dia *t*, a população inclui só quem **não** cancelou até *t*
  (`date_unregistration` ausente ou > *t*). 2.678 cancelamentos (26% dos
  `Withdrawn`) ocorrem antes do dia 0 e nunca entram em população de previsão.
- Cortes de referência analisados: dias 14, 28 e 42.

### TD-14 — Notas de avaliação como informação de disponibilidade incerta
- A OULAD registra a data de entrega, não a de divulgação da nota. Em fases
  futuras, uma nota só poderá ser usada com `date_submitted ≤ corte` e a
  aproximação deverá ser declarada (ou testada com uma defasagem adicional).

### TD-15 — Fase 02 em branch própria (`feat/data-exploration`)
- Criada a partir da `main` com a Fase 01 já integrada (commit `a4296d6`). Sem merge
  automático na `main`.

### TD-16 — Convenção temporal
- Dia 0 = início da apresentação; semana *k* = dias 7(k−1)…7k−1; previsão ao fim da
  semana *k* ⇒ `cutoff_day = 7k − 1` (S2 = 13, S4 = 27, S6 = 41). A Fase 01 usou os dias
  14/28/42 de forma provisória; os números da Fase 02 substituem aqueles.
- População do corte: matriculado até o corte e não cancelado até o corte.

### TD-17 — Agregação *as-of* em `src/feature_engineering/temporal.py`
- Colocada em `feature_engineering` porque será reutilizada na Fase 03. Nesta fase é
  usada só de forma descritiva. Testes garantem invariância a eventos após o corte.

### TD-18 — Novo pacote `src/analysis/`
- EDA, auditoria de vazamento e gráficos não pertencem a `data/` nem a `evaluation/`
  (que será de avaliação de modelos). Notebooks são finos e chamam `src/`.

### TD-19 — "Força de separação" = |AUC − 0,5| × 2 univariada
- Estatística descritiva sem treino, comparável entre variáveis. Não usada em variáveis
  nominais codificadas como inteiros (ordem arbitrária) — para estas, Cramér's V.
- `univariate_auc` recusa séries com índices diferentes (um bug desse tipo foi
  encontrado e corrigido durante a fase).

### TD-20 — Notebooks gerados e executados por script
- `python -m src.analysis.build_notebooks` gera e executa os 3 notebooks (saídas
  versionadas para leitura no GitHub; contêm só agregados).

### TD-21 — Artefatos versionados × não versionados
- Versionados: `research/figures/*.png`, `research/inspection/*.json|csv` (agregados).
- Não versionados: `data/interim/*.csv.gz` (snapshots por estudante, ~MB).

### TD-22 — Dependências adicionadas
- `matplotlib`, `seaborn`, `nbformat`, `nbconvert`, `ipykernel` (versões fixadas).

### TD-23 — Decisões metodológicas aprovadas (Fase 03)
- OULAD principal: `target_withdrawn` (Withdrawn = 1; Pass, Distinction, Fail = 0).
  Secundário: `target_adverse_outcome` (Withdrawn, Fail = 1). Resolve Q1b.
- UCI: Dropout = 1, Graduate = 0, Enrolled excluído (guardado sem rótulo). Resolve Q5.
- Cortes S2 (alerta precoce), S4 (principal), S6 (comparativo). Resolve Q6.

### TD-24 — Notas em quarentena (resolve Q12)
- Nenhuma feature usa `score`. O sinal acadêmico vem de entregas, atrasos e não entregas.

### TD-25 — Variáveis condicionais fora do conjunto principal (resolve Q13)
- OULAD: `studied_credits`. UCI: `Scholarship holder`, `Debtor`, `Tuition fees up to date`,
  inscrições/créditos do 1º semestre e macro (estas últimas viram `cohort_proxy`, só auditoria).

### TD-26 — Resultados *banked*
- Todas as 1.909 linhas têm `date_submitted = −1`. Não contam como entrega nem nota; a
  avaliação sai do denominador de vencidas do estudante.

### TD-27 — Rótulos inconsistentes excluídos
- 93 `Withdrawn` sem data e 9 não-`Withdrawn` com data saem de todos os cortes.

### TD-28 — Linhas duplicadas do AVA mantidas (Q11)
- Cliques somados por estudante/dia, sem deduplicar (decisão conservadora; pendente de revisão).

### TD-29 — Parquet + manifestos versionados
- Parquet (pyarrow 18.1.0) em `data/processed/` não é versionado. `*.metadata.json` e
  `*.manifest.json` são versionados (exceção no `.gitignore`).
- Hash de conteúdo calculado sobre os dados relidos do disco; hashes dos brutos são
  verificados a cada construção.

### TD-30 — Tipos canônicos e sem transformações globais
- Tipo de cada feature vem do catálogo. Nenhuma imputação, codificação, normalização ou
  balanceamento nesta fase: tudo será ajustado só no treino (Fase 04).

### TD-31 — IDs da OULAD mantidos
- `id_student` já é pseudonimizado pela Open University; mantido como `student_id` para
  validação por grupo. UCI não tem ID: `row_id` = posição no arquivo original.

### TD-32 — Validação (Q15, decisão aprovada)
- Treino 2013 e teste 2014 sem os estudantes presentes em 2013 (1.368 em S4). Os recorrentes
  ficam num cenário secundário, reportado à parte.
- Dentro de 2013, divisão por estudante: 60% treino, 20% validação (seleção) e 20% calibração
  (calibrador, threshold, faixas). A terceira partição evita ajustar calibração e threshold nos
  dados usados para escolher o modelo.

### TD-33 — Gênero e deficiência fora dos preditores (Q16, decisão aprovada)
- `EXCLUDED_SENSITIVE` = {adm_gender, adm_disability}. Na UCI, por coerência, também `Gender` e
  `Educational special needs` (análogo a deficiência), embora façam parte das 18 features de
  ingresso aprovadas. Todos são usados só na auditoria.

### TD-34 — Modelo baseado em árvores: HistGradientBoosting
- Não linearidades e interações, NaN nativo, rapidez, regularização controlável e probabilidades
  em geral melhores que Random Forest. Early stopping desligado (dividiria o treino por linha,
  não por estudante).

### TD-35 — Indicadores de ausência seletivos
- Só para `asm_submission_rate`, `asm_mean_days_before_deadline` e `adm_imd_band`. Nunca para
  `adm_registration_lead_days` (quarentena).

### TD-36 — Calibração conservadora
- none/Platt/isotônico escolhido pelo Brier fora do fold dentro da partição de calibração; troca
  "none" só com ganho > 0,5%. Na S4 nenhum calibrador foi necessário.

### TD-37 — Faixas de risco preliminares
- Baixo < prevalência do treino ≤ moderado < máx(threshold F1, 2 × prevalência) ≤ alto.
  Definidas só com dados de 2013. Ver `docs/RISK_CLASSIFICATION.md`.

### TD-38 — Balanceamento não adotado
- `class_weight="balanced"` testado na S4: mesma PR-AUC, calibração muito pior. Mantida a
  distribuição natural; nada de SMOTE.

### TD-39 — Validação da UCI sem simulação temporal
- Hold-out estratificado de 20% + validação cruzada estratificada de 5 folds; deixar uma coorte
  de fora apenas como teste de robustez (não temporal).

### TD-40 — Nominais inteiros convertidos em texto
- Os códigos nominais da UCI (inteiros) são convertidos para texto antes do one-hot (erro
  detectado na primeira execução). Para a OULAD a conversão não altera nada: os resultados das
  duas execuções foram idênticos.

### TD-41 — Artefatos
- Versionados: `reports/modeling/**/results.json`, CSVs de comparação e PNGs. Não versionados:
  `models.joblib` e `test_predictions.parquet`.

## Questões pendentes

| # | Questão | Por que importa |
|---|---|---|
| Q1 | ~~Autorizar download da OULAD~~ — **resolvido**: autorizado e baixado em 2026-11-05 | — |
| Q1b | OULAD: `Fail` conta como 0, é excluído (opção A) ou entra como positivo (opção C)? Fase 02 mostrou que 41,7% dos Fail abandonaram o AVA antes da metade do módulo | Define o rótulo |
| Q1c | OULAD: quem nunca acessou o AVA até o corte (921 no dia 28) — tratar como sinal de risco ou como caso de abandono já ocorrido de fato? | Pode inflar o desempenho de forma artificial |
| Q2 | Autorizar download parcial da **UFRN** (quais anos/níveis)? | Única fonte brasileira individual encontrada |
| Q3 | Termos exatos da licença "Outra (Aberta)" da UFRN | Uso e redistribuição |
| Q4 | Dicionário de situações da UFRN: quais valores = evasão (CANCELADO? por abandono, transferência, desligamento?) | Definição do rótulo |
| Q5 | Classe `Enrolled` da UCI: proposta = excluir no rótulo principal (A) e usar B como sensibilidade | Ruído de rótulo |
| Q6 | Corte: proposta = semana 4 principal + semana 2 como alerta precoce (ver TEMPORAL_FEASIBILITY.md) | Define os cortes |
| Q7 | Qual será a fonte de dados **real** em produção (sistema acadêmico da IES parceira, AVA, frequência)? | O modelo final deve ser treinado com dados do mesmo tipo dos de produção |
| Q8 | Limiares baixo/moderado/alto | Depende da capacidade de intervenção da equipe |
| Q9 | Evasão de **módulo** (OULAD) vs. evasão de **curso** — aceitável como proxy metodológico? | Validade externa |
| Q10 | Verificar se o hash de estudante é o mesmo entre conjuntos da UFRN | Necessário para juntar situação + notas |
| Q11 | `studentVle`: 787.170 linhas totalmente duplicadas — manter (sessões distintas) ou deduplicar? | Afeta contagem de cliques |
| Q12 | Notas OULAD: aceitar "conhecida na entrega" ou aplicar defasagem fixa (+7/+14 dias)? | Otimismo das features acadêmicas |
| Q13 | Variáveis financeiras da UCI (`Tuition fees up to date`, `Debtor`, `Scholarship holder`): manter em quarentena? | Risco de vazamento |
| Q14 | Modelo de "dia 0" para os 26% que cancelam antes do início? | Escopo |
| Q15 | ~~Validação~~ **resolvida (TD-32)** — Validação na Fase 04: treinar 2013 → testar 2014 **e** agrupar por estudante (1.368 repetidos entre anos) | Evita vazamento entre treino e teste |
| Q16 | ~~Sensíveis~~ **resolvida (TD-33)** — Uso de `adm_gender` e `adm_disability` (sensíveis) no modelo ou só na auditoria de equidade? | Equidade |
| Q17 | Retirar `adm_imd_band` do modelo principal? Custo de PR-AUC de 0,003; com ela há excesso de falsos positivos nas áreas mais pobres (ver FAIRNESS_AUDIT.md) | Equidade |
| Q18 | Como tratar a subestimação do risco de estudantes com deficiência (−5 a −8 p.p.) sem usar a variável como preditora? | Equidade / calibração |
| Q19 | Threshold e faixas definidos junto com a equipe pedagógica, conforme a capacidade de atendimento | Uso institucional |
| Q20 | Validação com dados brasileiros (UFRN) antes de qualquer uso real | Validade externa |
