# Auditoria de Vazamento de Dados (Data Leakage)

> Fase 02 · 2026-11-05. Fonte única do catálogo: `src/analysis/leakage.py`
> (exportado em `research/inspection/leakage_catalog.json`). Nenhuma variável foi
> excluída definitivamente: o catálogo registra a classificação e o motivo.

**Classes**

- **PERMITIDA** — disponível antes ou no momento da previsão.
- **CONDICIONAL** — utilizável apenas com filtro temporal ou premissa documentada.
- **PROIBIDA** — usa informação futura ou revela diretamente o resultado.

> **Atualização da Fase 03 (decisões aprovadas):** notas (`score`) ficam **fora** dos
> conjuntos principais até existir evidência da data de divulgação; variáveis CONDICIONAIS
> sem data comprovada (`studied_credits`; na UCI `Scholarship holder`, `Debtor`,
> `Tuition fees up to date`, inscrições do 1º semestre e variáveis macro) também ficam fora.
> Ver `docs/FEATURE_ENGINEERING.md`.

## 1. Evidência empírica de vazamento

### OULAD — mesma população (semana 4), com e sem futuro

| Indicador | Força de separação |
|---|---|
| Cliques acumulados até o dia 27 (PERMITIDO com filtro) | 0,244 |
| Avaliações entregues até o dia 27 (PERMITIDO com filtro) | 0,133 |
| Cliques no **módulo inteiro** (PROIBIDO) | 0,679 |
| Avaliações entregues no **módulo inteiro** (PROIBIDO) | 0,901 |
| Último dia ativo no **módulo inteiro** (PROIBIDO) | **0,982** |

O "último dia ativo" quase reconstrói a data do cancelamento: um modelo com essa
variável pareceria excelente e seria inútil em produção.

### UCI — regras que revelam o desfecho

| Condição | n | % Dropout |
|---|---|---|
| 1º sem.: UCs inscritas > 0 e 0 avaliações | 169 | **100,0%** |
| 2º sem.: UCs inscritas > 0 e 0 avaliações | 221 | **99,5%** |
| Mensalidades **não** em dia | 528 | 86,6% |
| 2º sem.: 0 aprovações | 870 | 83,6% |
| 1º sem.: 0 aprovações | 718 | 79,4% |
| Devedor | 503 | 62,0% |
| 2º sem.: 0 UCs inscritas | 180 | 42,8% (não revela) |

## 2. OULAD — catálogo

| Tabela | Variável | Classe | Justificativa |
|---|---|---|---|
| studentInfo | code_module / code_presentation | PERMITIDA | Conhecidos na matrícula; usar para contexto/normalização |
| studentInfo | gender | PERMITIDA | Declarado na matrícula; requer revisão de equidade |
| studentInfo | region | PERMITIDA | Declarado na matrícula |
| studentInfo | highest_education | PERMITIDA | Escolaridade prévia |
| studentInfo | imd_band | PERMITIDA | Índice de privação do endereço; 1.111 ausentes |
| studentInfo | age_band | PERMITIDA | Conhecido na matrícula |
| studentInfo | disability | PERMITIDA | Declarado; dado sensível — revisão de equidade obrigatória |
| studentInfo | num_of_prev_attempts | PERMITIDA | Conta tentativas **passadas** |
| studentInfo | studied_credits | CONDICIONAL | Data do registro não documentada; pode incluir módulos adicionados/retirados no ano |
| studentInfo | final_result | **PROIBIDA** | É o desfecho |
| studentRegistration | date_registration | PERMITIDA | Conhecida a partir da matrícula |
| studentRegistration | date_registration **ausente** | CONDICIONAL | 45 casos, 86,7% Withdrawn: a ausência pode ser efeito do cancelamento. Não usar indicador de ausência |
| studentRegistration | date_unregistration | **PROIBIDA** como feature | Define o evento. Uso permitido **somente** para definir a população do corte |
| courses | module_presentation_length | PERMITIDA | Planejado e publicado |
| assessments | assessment_type, weight, date (prazo) | PERMITIDA | Calendário publicado; 11 prazos de prova ausentes |
| vle | activity_type, week_from, week_to | PERMITIDA | Desenho do curso (82% de semanas ausentes) |
| studentVle | sum_click, id_site, date com `date ≤ corte` | CONDICIONAL | Cada interação tem data: uso só até o corte |
| studentVle | qualquer agregado do período inteiro | **PROIBIDA** | Inclui o futuro e a ausência de cliques após o cancelamento |
| studentAssessment | date_submitted ≤ corte | CONDICIONAL | Só entregas até o corte |
| studentAssessment | score (entrega ≤ corte) | CONDICIONAL | Sem data de divulgação; assumir "conhecida na entrega" é otimista — testar defasagem de +7/+14 dias |
| studentAssessment | is_banked = 1 | CONDICIONAL | Nota aproveitada de apresentação anterior; semântica de data não documentada; excluída por ora |
| studentAssessment | nota da prova final | **PROIBIDA** | Prova ocorre no fim |
| derivada | contagem/média de avaliações do período inteiro | **PROIBIDA** | Entregas ausentes após cancelamento codificam o rótulo |
| derivada | taxa de evasão da **mesma** apresentação | **PROIBIDA** | Codificação do alvo com os rótulos da coorte avaliada |
| derivada | estatísticas de apresentações **anteriores** | CONDICIONAL | Só se calculadas com apresentações já encerradas |
| derivada | normalização de cliques por módulo | CONDICIONAL | Só com dados ≤ corte e ajustada no conjunto de treino |
| desenho | mesmo `id_student` em várias apresentações (3.538) | CONDICIONAL | Separar treino/teste por estudante e/ou por tempo |

## 3. UCI 697 — catálogo por cenário

| Variável | Semanas 2–6 do 1º sem. | Semanas 2–6 do 2º sem. | Justificativa |
|---|---|---|---|
| Marital status, Application mode, Application order, Course, Daytime/evening attendance, Previous qualification, Previous qualification (grade), Nacionality, Mother's/Father's qualification, Mother's/Father's occupation, Admission grade, Displaced, Educational special needs, Gender, Age at enrollment, International | PERMITIDA | PERMITIDA | Documentadas como conhecidas no ingresso |
| Debtor | CONDICIONAL | CONDICIONAL | Data de coleta não documentada; pode refletir dívida posterior ao abandono |
| Tuition fees up to date | CONDICIONAL | CONDICIONAL | Idem; separação 0,31 e 86,6% de Dropout entre os "não em dia" — **maior risco** do grupo |
| Scholarship holder | CONDICIONAL | CONDICIONAL | Bolsas podem ser concedidas/perdidas durante o curso |
| Curricular units 1st sem (enrolled / credited) | CONDICIONAL | PERMITIDA | Definidas na inscrição do semestre; momento da extração não documentado |
| Curricular units 1st sem (evaluations / approved / grade / without evaluations) | **PROIBIDA** | PERMITIDA | Fim do 1º semestre. "Inscrito e 0 avaliações" = 100% Dropout: registra o abandono |
| Curricular units 2nd sem (enrolled / credited) | **PROIBIDA** | CONDICIONAL | Conhecidas no início do 2º sem.; podem refletir saída já ocorrida |
| Curricular units 2nd sem (evaluations / approved / grade / without evaluations) | **PROIBIDA** | **PROIBIDA** | Fim do 2º semestre |
| Unemployment rate, Inflation rate, GDP | CONDICIONAL | CONDICIONAL | Ano de referência não documentado; só 10 valores → funcionam como identificador de coorte |
| Target | **PROIBIDA** | **PROIBIDA** | Rótulo |

## 4. Riscos metodológicos identificados

1. **Agregados do período inteiro** (OULAD) — o risco mais grave; separação até 0,98.
2. **População do corte mal definida** — incluir quem já cancelou transforma "prever"
   em "constatar". Implementado em `population_at_cutoff` e testado.
3. **Notas sem data de divulgação** (OULAD) — usar a data de entrega é otimista.
4. **Ausência de datas de abandono** (UCI) — impede definir a população no 2º semestre.
5. **Variáveis financeiras sem data** (UCI) — forte associação compatível com medição
   pós-abandono.
6. **Mesmo estudante em várias apresentações** (OULAD) — vazamento entre treino e teste.
7. **Codificação do alvo e normalizações** com dados da própria coorte de teste.
8. **Variáveis macroeconômicas como identificador de coorte** (UCI) — o modelo pode
   aprender a coorte em vez do estudante.
9. **Exclusão de grupos ambíguos** (`Enrolled`, `Fail`) — melhora métricas artificialmente
   se a avaliação final não incluir toda a população do corte.
10. **Validação aleatória** — deve ser temporal (ex.: treinar em 2013, testar em 2014) e/ou
    por grupo, nunca uma divisão aleatória de linhas.
