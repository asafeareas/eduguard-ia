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
