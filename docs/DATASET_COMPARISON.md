# Comparação entre Datasets — OULAD × UCI 697

> Fase 02 · 2026-11-05. Base: análises em `docs/EXPLORATORY_DATA_ANALYSIS.md`,
> `docs/TEMPORAL_FEASIBILITY.md` e `docs/DATA_LEAKAGE_AUDIT.md`.

| # | Critério | OULAD | UCI 697 |
|---|---|---|---|
| 1 | **Granularidade** | Estudante × módulo × apresentação (32.593), com tabelas de eventos diários (10,7 M cliques; 173.912 entregas) | Um registro por estudante (4.424), já agregado por semestre |
| 2 | **Qualidade** | Chaves íntegras (0 órfãos); missing `"?"`; 7,4% de linhas duplicadas no AVA; 93+9 inconsistências de rótulo; 1.111 IMD ausentes | Sem ausentes ou duplicatas; consistente; cabeçalho com TAB e BOM; nominais em código numérico; metadado de tipo incorreto |
| 3 | **Rótulos** | `final_result` (4 classes) + **data** do cancelamento | `Target` (3 classes), **sem data**; `Enrolled` censurado |
| 4 | **Temporalidade** | Diária, relativa ao início do módulo | Apenas ingresso, fim do 1º e fim do 2º semestre |
| 5 | **Variáveis acadêmicas** | Entregas, atrasos e notas por avaliação, com data; prova final | UCs inscritas/avaliadas/aprovadas e média por semestre; nota de admissão; escolaridade prévia |
| 6 | **Variáveis comportamentais** | Cliques diários por material (20 tipos) | **Nenhuma** |
| 7 | **Previsão antecipada** | **Sim** — cortes reais nas semanas 2/4/6 com população correta | Semanas 2–6 do 1º sem.: só ingresso (sinal fraco). 2º sem.: sinal forte, mas sem como excluir quem já saiu |
| 8 | **Limitações** | EAD britânico; módulo ≠ curso; sem frequência; nota sem data de divulgação; 2013–2014 | Uma instituição portuguesa; sem datas; variáveis financeiras sem data; coorte só por proxy |
| 9 | **Compatibilidade com o EduGuard** | **Alta** para validar a mecânica de alerta semanal | **Parcial**: ingresso e histórico, sem dimensão semanal |
| 10 | **Uso conjunto** | Não recomendado | Não recomendado |

## Por que **não** integrar

1. **Unidades diferentes**: módulo de ~9 meses (OULAD) × curso completo (UCI).
2. **Rótulos diferentes**: `Withdrawn` é cancelamento formal de um módulo; `Dropout` é
   abandono do curso medido ao fim da duração normal.
3. **Populações diferentes**: EAD britânico adulto (OULAD) × politécnico presencial
   português, majoritariamente jovem (UCI).
4. **Quase nenhuma variável em comum**: só faixa etária, gênero e escolaridade prévia — e
   com codificações diferentes. Engajamento (OULAD) e desempenho semestral (UCI) não têm
   equivalente na outra base.
5. Concatenar criaria um conjunto artificial em que o modelo aprenderia a **identificar a
   base de origem**, não o risco do estudante.

## Integração admissível (futura e limitada)

- **Comparar protocolos, não dados:** usar o mesmo pipeline de avaliação (cortes, métricas,
  calibração, equidade) nas duas bases e comparar conclusões.
- **Transferir hipóteses:** por exemplo, "idade e escolaridade prévia têm associação
  fraca e consistente nas duas bases" — verificado separadamente em cada uma.
- Nunca misturar estudantes, instituições ou períodos num único conjunto de treino.

## Recomendação

| Papel | Dataset |
|---|---|
| **Principal** para desenvolver e validar a previsão nas primeiras semanas | **OULAD** (corte principal: semana 4; alerta precoce: semana 2) |
| Secundário: modelo de **ingresso** (dia 0) e referência comparável à literatura | **UCI 697**, rótulo A (Dropout × Graduate), só variáveis PERMITIDAS no cenário do 1º semestre |
| Validação no contexto brasileiro (pendente de autorização) | **UFRN Dados Abertos** — notas por unidade e faltas; único com frequência |

Os resultados da OULAD **não devem ser apresentados** como desempenho esperado em uma
instituição brasileira presencial sem validação com dados nacionais.
