# Pesquisa de Datasets — EduGuard-IA (Fase 01)

> Data da pesquisa: **2026-11-05**. Todas as informações abaixo foram verificadas
> nas páginas oficiais nesta data, salvo quando marcadas como **[não verificado]**.
> Onde um dado não pôde ser confirmado, ele aparece como **desconhecido** — nada foi
> preenchido por suposição.

## 1. Objetivo da pesquisa

Encontrar bases que permitam treinar um classificador supervisionado de evasão
(`EVADIU = 1`, `NÃO EVADIU = 0`) **usando apenas informações disponíveis nas
primeiras semanas do semestre**. O critério decisivo, portanto, não é "ter a
palavra *dropout*", mas sim:

1. dados **individualizados** (um registro por estudante);
2. **rótulo de desfecho** observado depois do momento da previsão;
3. **marcação temporal** suficiente para reconstruir o que era conhecido no
   corte de previsão (ex.: semana 2, 4 ou 6 do semestre).

## 2. Quadro comparativo

| # | Dataset | País | Nível | Rótulo de evasão | Temporalidade intra-semestre | Frequência | Engajamento digital | Licença | Compatibilidade |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **OULAD** (Open University) — **baixado** | Reino Unido | Individual (pseudonimizado) | Sim (`Withdrawn` + `date_unregistration`) | **Sim — dia relativo ao início do módulo** | Não (EAD) | **Sim — cliques diários no AVA** | CC BY 4.0 | **ALTA** |
| 2 | **UFRN Dados Abertos** (situações + matrículas em componentes) | Brasil | Individual (ID com hash) | Sim (situação: `CANCELADO`, `CONCLUÍDO`, …) com data da alteração | Parcial — notas por **unidade avaliativa** (1ª, 2ª, 3ª), sem data | **Sim — faltas** | Não encontrado | "Outra (Aberta)" — termos exatos **não verificados** | **ALTA (condicionada)** |
| 3 | **Predict Students' Dropout and Academic Success** (UCI 697) | Portugal | Individual | Sim (`Dropout`/`Enrolled`/`Graduate`) | **Não** — apenas fim do 1º e do 2º semestre | Não | Não | CC BY 4.0 | **PARCIAL** |
| 4 | Microdados do **Censo da Educação Superior** (INEP) | Brasil | **Agregado** por curso/IES (versão pública atual) | Não por estudante | Não (anual) | Não | Não | Dados abertos gov. | **BAIXA** |
| 5 | **Indicadores de Fluxo da Educação Superior** (INEP) | Brasil | **Agregado** (curso × coorte) | Taxas (TAP, TCA, TDA, TCAN, TADA) | Não (anual) | Não | Não | Dados abertos gov. | **BAIXA** (útil como contexto) |
| 6 | **Plataforma Nilo Peçanha** (MEC/Rede Federal) | Brasil | **desconhecido** | Indicadores de evasão (nível de agregação **não verificado**) | Não verificado | Não verificado | Não | CC BY-ND 3.0 (site gov.br) | **BAIXA / não confirmada** |
| 7 | KDD Cup 2015 (XuetangX MOOC) | China | Individual | "Dropout" = sem atividade nos 10 dias seguintes | Sim (logs) | Não | Sim | **desconhecida** | **BAIXA** (evasão de MOOC ≠ evasão de curso) |
| 8 | xAPI-Edu-Data (Kalboard 360, Kaggle) | Jordânia | Individual | **Não** (classe de desempenho L/M/H) | Não | Ausências (2 níveis) | Sim (agregados) | **desconhecida** | **BAIXA** |
| 9 | Student Performance (UCI 320) **[não verificado]** | Portugal | Individual (ensino médio) | **Não** | Notas G1/G2/G3 | Ausências | Não | CC BY 4.0 [não verificado] | **BAIXA** |

## 3. Fichas detalhadas

### 3.1 OULAD — Open University Learning Analytics Dataset — **ALTA COMPATIBILIDADE**

| Critério | Informação |
|---|---|
| Instituição | The Open University (Knowledge Media Institute) — Kuzilek, Hlosta, Zdrahal |
| URL oficial | https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/ · UCI: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset |
| País | Reino Unido (ensino a distância) |
| Registros | 32.593 matrículas estudante×módulo (28.785 estudantes), 22 apresentações de 7 módulos — confirmado após download |
| Variáveis / tabelas | 7 CSVs: `courses`, `assessments`, `vle`, `studentInfo`, `studentRegistration`, `studentAssessment`, `studentVle` |
| Formato | CSV dentro de ZIP — **44,6 MB** compactado; `studentVle.csv` sozinho tem **432,8 MB** |
| Licença | CC BY 4.0 (citar Kuzilek et al., 2017, *Scientific Data* 4:170171) |
| Individual/agregado | Individual, pseudonimizado (`id_student`) |
| Rótulo | `final_result` ∈ {Pass, Fail, Withdrawn, Distinction} (confirmado); `date_unregistration` (dia do cancelamento) |
| Acadêmicas | Notas e datas de entrega de avaliações (TMA, CMA, Exame) |
| Frequência | Não há presença física (EAD) |
| Engajamento digital | **Sim** — `sum_click` por dia por recurso do AVA |
| Temporalidade | **Sim** — todas as datas em dias relativos ao início do módulo (dia 0) |
| Primeiras semanas | **Reconstruível**: corte em dia 14/28/42, usando só cliques e avaliações com `date ≤ corte` |
| Limitações | (a) Unidade é o **módulo** (~9 meses), não o curso inteiro: `Withdrawn` é cancelamento do módulo, não necessariamente evasão do curso; (b) contexto britânico de EAD, distante do presencial brasileiro; (c) estudantes que cancelaram **antes** do corte precisam ser excluídos do conjunto de previsão daquele corte (não há o que prever); (d) `imd_band` com valores ausentes (UCI: "Has Missing Values? Yes") |

**Por que é a mais adequada:** é a única base verificada que permite simular
literalmente "o que o sistema sabia na semana *k*", porque cada evento possui data.

#### Verificação após download (2026-11-05) — `research/inspection/oulad_349_profile.json`

- **32.593** matrículas estudante×apresentação, **28.785** estudantes distintos,
  22 apresentações de 7 módulos (234–269 dias). `studentVle`: **10.655.280** linhas,
  datas de −25 a 269.
- `final_result`: Pass 12.361 (37,9%) · **Withdrawn 10.156 (31,2%)** · Fail 7.052
  (21,6%) · Distinction 3.024 (9,3%).
- Valores ausentes codificados como `"?"` nos CSVs; `imd_band` tem 1.111 ausentes,
  `date_registration` 45, `score` 173, `assessments.date` 11.
- **Achado metodológico central — 26% dos cancelamentos ocorrem antes do início:**
  2.678 matrículas têm `date_unregistration < 0` (mediana dos cancelamentos = dia
  27; quartil inferior = dia −2). Esses casos não são "previsíveis durante o
  semestre" e devem sair da população de cada corte.
- **População e rótulo por corte** (excluindo quem já cancelou até o corte):

  | Corte | População | Withdrawn restantes | Sem nenhum clique até o corte | Mediana de cliques Withdrawn × Pass |
  |---|---|---|---|---|
  | dia 14 | 28.119 | 5.690 (20,2%) | 1.691 | 90 × 159 |
  | dia 28 | 27.538 | 5.109 (18,6%) | 921 | 183 × 280 |
  | dia 42 | 27.024 | 4.595 (17,0%) | 717 | 235 × 365 |

  Há sinal de engajamento já na semana 2, sem usar informação futura.
- Primeira avaliação não-final vence, por apresentação, entre o dia 12 e o 61
  (mediana 21,5): no corte do dia 28 parte das apresentações já tem uma nota. A
  **data de divulgação da nota não existe** — usar `date_submitted ≤ corte` é uma
  aproximação a documentar.
- Inconsistências: 93 `Withdrawn` sem `date_unregistration`; 9 não-`Withdrawn`
  com `date_unregistration`; 3.365 matrículas sem nenhum registro no AVA;
  `date_submitted` chega a 608 (maior que a duração do módulo; provável `is_banked`).

### 3.2 UFRN — Portal de Dados Abertos — **ALTA COMPATIBILIDADE (condicionada)**

Fonte brasileira, individualizada, com histórico semestral. Conjuntos relevantes
(API CKAN `https://dados.ufrn.br/api/3/action/package_show?id=<id>`):

| Conjunto | Conteúdo verificado (cabeçalhos lidos em 2026-11-05) | Cobertura |
|---|---|---|
| `situacoes-de-discentes` | `id_discente` (hash), `ano_ingresso`, `periodo_ingresso`, `nivel_ensino`, `id_curso`, `curso`, `situacao`, `data_alteracao_situacao`, `ano_alteracao_situacao`, `periodo_alteracao_situacao` | 2009.1 → 2026.2 (37 recursos incl. dicionário PDF) |
| `matriculas-componentes` | `id_turma`, `discente` (hash), `id_curso`, `unidade`, `nota`, `reposicao`, `faltas_unidade`, `media_final`, `numero_total_faltas`, `descricao` (ex.: `APROVADO POR NOTA`, `REPROVADO POR MÉDIA E POR FALTAS`, `CANCELADO`) | ~43 arquivos semestrais, **~65–77 MB cada** (2019.1: 76,5 MB; 2024.2: 66,5 MB) |
| `turmas` | Relação de turmas (para ligar `id_turma` a componente e período) | 55 recursos — colunas **não verificadas** |
| `dados-socio-economicos-de-discentes` | Dados socioeconômicos por período (IDs mascarados) | 2018.2 → 2021.2 (aprox.) — colunas **não verificadas** |
| ⚠️ `discentes` | **Contém nome completo e número de matrícula reais** | **NÃO UTILIZAR** (dado pessoal identificável) |

| Critério | Informação |
|---|---|
| Licença | Metadado CKAN: "Outra (Aberta)" (`other-open`). Termos exatos **não verificados** |
| Rótulo | `situacao` com data — permite rotular evasão (`CANCELADO`, possivelmente outros estados) vs. `CONCLUÍDO`/`ATIVO`. O significado exato de cada situação precisa ser confirmado no **dicionário de dados** (PDF) |
| Frequência | **Sim** — `faltas_unidade`, `numero_total_faltas` |
| Engajamento digital | Não encontrado nos dados abertos |
| Primeiras semanas | **Parcial**: as notas da **1ª unidade** existem, mas sem data de lançamento; a 1ª unidade costuma encerrar após as primeiras semanas. Para previsão no **início do 2º semestre em diante**, o histórico dos semestres anteriores é totalmente utilizável e não vaza informação futura |
| Limitações | (a) Dados pseudonimizados continuam sendo dados pessoais (LGPD) — proibido tentar reidentificar; (b) consistência do hash entre conjuntos (`id_discente` vs. `discente`) **não verificada**; (c) volume total ~3 GB; (d) período 2020–2021 (pandemia) atípico; (e) uma única instituição |

### 3.3 UCI 697 — Predict Students' Dropout and Academic Success — **COMPATIBILIDADE PARCIAL** (baixado)

| Critério | Informação |
|---|---|
| Instituição | Instituto Politécnico de Portalegre (Realinho, Vieira Martins, Machado, Baptista) |
| URL | https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success · DOI 10.24432/C5MC89 |
| País | Portugal |
| Registros / variáveis | **4.424** estudantes · **36 atributos + 1 alvo** (confirmado no arquivo) |
| Formato | `data.csv` (separador `;`, UTF-8 com BOM), 533.230 bytes |
| Licença | CC BY 4.0 |
| Rótulo | `Target`: Graduate 2.209 (49,9%) · Dropout 1.421 (32,1%) · Enrolled 794 (17,9%), medido "ao final da duração normal do curso" |
| Acadêmicas | Unidades curriculares inscritas/avaliadas/aprovadas/média no **1º e 2º semestres** |
| Frequência / engajamento | **Não existem** |
| Temporalidade | Apenas dois blocos: matrícula e **fim** do 1º/2º semestre. **Não há ano de ingresso**; as variáveis macro (desemprego, inflação, PIB) têm exatamente 10 combinações distintas, o que sugere 10 coortes — **inferência, não documentada** |

**Avaliação de previsão antecipada (pergunta central do briefing):**

- **Semanas iniciais do 1º semestre:** só as variáveis de matrícula estariam
  disponíveis (demográficas, socioeconômicas, modo de candidatura, nota de
  admissão). É possível modelar, mas sem nenhum sinal comportamental do semestre.
- **Variáveis do 1º/2º semestre:** são resultados **do fim** do semestre. Usá-las
  para prever no início do mesmo semestre é **vazamento de informação futura**.
  Constatação objetiva: os **169** estudantes com unidades inscritas e **zero
  avaliações no 1º semestre são 100% `Dropout`** — ou seja, a variável já
  registra o abandono, em vez de antecipá-lo.
- **`Debtor`, `Tuition fees up to date`, `Scholarship holder`:** o momento de
  coleta **não é documentado**. `Tuition fees up to date = 0` ocorre em 32,2% dos
  Dropout contra 1,3% dos Graduate — sinal forte, mas possivelmente medido depois
  do abandono. Tratar como **suspeita de vazamento** até prova em contrário.
- **Cenário válido:** prever no **início do 2º semestre** usando matrícula +
  resultados do 1º semestre (excluindo variáveis do 2º semestre e as financeiras
  de data incerta).
- **Classe `Enrolled`:** não é nem evasão nem conclusão; o desfecho é censurado.
  Mapeá-la para 0 introduz ruído de rótulo — decisão pendente (ver
  TECHNICAL_DECISIONS.md).

### 3.4 INEP — Microdados do Censo da Educação Superior — **BAIXA COMPATIBILIDADE**

- URL: https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/censo-da-educacao-superior
  (padrão de download `https://download.inep.gov.br/microdados/microdados_censo_da_educacao_superior_AAAA.zip`, 1995–2024).
- Em 2022 o INEP retirou as bases antigas do portal para "adequação à LGPD" e
  passou a publicar arquivos **agregados por curso/IES**; os campos de aluno
  (CPF, código do aluno etc.) não aparecem nas versões públicas.
- O portal informa que a Plataforma **SEDAP+** (lançada em 28/04/2026) opera
  "exclusivamente sobre dados agregados", com ruído de privacidade diferencial;
  e que uma Ação Civil Pública transitou em julgado em 20/03/2026 sem obrigar o
  retorno ao formato anterior a 2020 (extraído de resumo da página — confirmar
  nos PDFs oficiais).
- **Não há trajetória individual nem rótulo por estudante na versão pública.**
- Tamanho do ZIP 2024: **desconhecido** (o servidor encerrou a conexão durante a
  verificação).
- **Uso possível:** variáveis contextuais por curso/IES (ex.: taxa de evasão
  histórica do curso), sempre com defasagem temporal correta.

### 3.5 INEP — Indicadores de Fluxo da Educação Superior — **BAIXA COMPATIBILIDADE**

- URL: https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/indicadores-educacionais/indicadores-de-fluxo-da-educacao-superior (período 2010–2024; publicado em 22/09/2025).
- Indicadores por **curso × coorte de ingresso**: Taxa de Permanência (TAP), Taxa
  de Conclusão Acumulada (TCA), Taxa de Desistência Acumulada (TDA), Taxa de
  Conclusão Anual (TCAN), Taxa de Desistência Anual (TADA) — conforme literatura
  que cita o INEP; as abas da página não carregaram na verificação automática,
  portanto formato e links dos arquivos são **desconhecidos**.
- Útil para **benchmark** e para contextualizar o risco por curso, não para treinar
  um classificador individual.

### 3.6 Plataforma Nilo Peçanha (MEC) — **BAIXA / não confirmada**

- Páginas gov.br (https://www.gov.br/mec/pt-br/pnp) listam edições Ano Base
  2017–2022 e um "Módulo Extrator de Dados PNP" (CSV). A literatura cita
  microdados de matrículas com "situação de matrícula" (evadidos, concluintes, em
  curso).
- **Não verificado:** granularidade real, presença de ID de estudante, colunas.
  O domínio `plataformanilopecanha.mec.gov.br` falhou no handshake TLS e
  `dadosabertos.mec.gov.br/pnp` retornou 403.
- Licença do site gov.br: CC BY-ND 3.0 (a cláusula "sem derivações" pode
  restringir redistribuição de dados transformados — avaliar).

### 3.7 Fontes internacionais de baixa compatibilidade

- **KDD Cup 2015 / XuetangX** — 79.186 usuários, 39 cursos (segundo literatura).
  "Evasão" definida como inatividade nos 10 dias seguintes em MOOC: conceito
  diferente de evasão de curso superior. Licença e acesso atual **desconhecidos**
  (biendata.xyz / moocdata.cn).
- **xAPI-Edu-Data** — 480 registros, 16 atributos, Universidade da Jordânia. Sem
  rótulo de evasão (classe de desempenho). Licença **desconhecida**.
- **Student Performance (UCI 320)** — ensino médio, notas G1/G2/G3. Sem rótulo de
  evasão. **[não verificado nesta sessão]**.

## 4. Conclusão metodológica

1. **Nenhuma base brasileira pública e oficial do INEP** permite treinar um modelo
   individual de evasão — a granularidade pública é agregada.
2. A **UFRN** é a melhor fonte brasileira encontrada: individual, com situação
   datada, notas por unidade e **faltas**. Precisa de autorização para download
   (volume e dados pseudonimizados) e confirmação do dicionário de dados.
3. A **OULAD** é a melhor base para validar a **mecânica** de previsão por
   semana (cortes temporais reais, engajamento digital).
4. A **UCI 697** serve como base de referência/comparação com a literatura, desde
   que restrita ao cenário "início do 2º semestre" e sem as variáveis de data
   incerta. **Não deve ser vendida como previsão nas primeiras semanas do 1º
   semestre.**
5. As bases **não devem ser fundidas**: populações, definições de evasão e
   escalas de nota são incompatíveis. Cada uma deve ser um experimento separado.

## 5. Fontes consultadas

- UCI 697: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success (API: https://archive.ics.uci.edu/api/dataset?id=697)
- UCI 349 (OULAD): https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
- OU Analyse: https://research.stem.open.ac.uk/ouanalyse/open-dataset-more/
- Kuzilek et al. (2017): https://doi.org/10.1038/sdata.2017.171
- INEP microdados: https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados
- INEP RIPD: https://download.inep.gov.br/microdados/RIPD_dos_microdados_dos_Censos_da_Educa%C3%A7%C3%A3o.pdf
- Iede sobre retirada de microdados: https://portaliede.org.br/contribuicao/inep-mantem-tendencia-e-exclui-dados-tambem-do-censo-da-educacao-superior/
- Indicadores de fluxo (literatura): https://periodicos.ufpb.br/index.php/rteo/article/view/57234
- UFRN CKAN: https://dados.ufrn.br/api/3/action/package_search?q=discentes
- PNP: https://www.gov.br/mec/pt-br/pnp
- KDD Cup 2015 (Zenodo): https://zenodo.org/record/5914059
- xAPI-Edu-Data: https://research.ju.edu.jo/research/groups/EVO-ML/Pages/Resources.aspx
