# Disponibilidade e Restrições dos Dados da UFRN

> Fase 05 · 2026-11-05. **Nenhum acesso novo foi feito.** Este documento consolida o que foi
> verificado na Fase 01 (metadados públicos do portal CKAN e cabeçalhos dos arquivos, em
> 2026-11-05) e registra as restrições.

## 1. Situação

| Item | Situação |
|---|---|
| Autorização institucional para uso em pesquisa | **Pendente** |
| Dados baixados pelo projeto | **Nenhum** |
| Uso nesta fase | **Nenhum** (proibido sem autorização) |

## 2. O que se sabe (Fase 01)

| Conjunto (portal dados.ufrn.br) | Conteúdo verificado | Volume | Observações |
|---|---|---|---|
| `situacoes-de-discentes` | situação do discente com **data de alteração**; ano/período de ingresso; curso | ~6,5 MB por período, 2009.1–2026.2 | IDs em *hash* |
| `matriculas-componentes` | nota e **faltas por unidade**, média final, situação no componente | ~65–77 MB por período (~3 GB no total) | sem datas de lançamento; IDs em *hash* |
| `turmas` | relação de turmas | 55 recursos | colunas **não verificadas** |
| `dados-socio-economicos-de-discentes` | dados socioeconômicos por período | — | colunas **não verificadas**; uso só para auditoria |
| `discentes` | ⚠️ **nome completo e número de matrícula reais** | — | **não usar** (dado pessoal identificável) |

Licença declarada no CKAN: "Outra (Aberta)" (`other-open`). Os termos exatos **não foram
verificados** (pendência Q3).

## 3. Restrições

1. **Autorização:** mesmo sendo públicos, os dados só serão usados com autorização institucional
   formal (decisão Q20), para garantir finalidade, governança e possibilidade de parceria
   (logs do AVA e dados com datas não estão no portal aberto).
2. **LGPD:** dados pseudonimizados continuam sendo dados pessoais. A lei só deixa de se aplicar
   a dados **anonimizados** (art. 12); a pseudonimização, definida no art. 13, §4º, mantém o
   vínculo com o titular. É proibido tentar reidentificar pessoas. O conjunto `discentes` está excluído.
3. **Ligação entre tabelas:** não está confirmado se os *hashes* `discente` e `id_discente`
   coincidem (Q10). Isso deve ser validado com a instituição, nunca por tentativa de cruzamento com dados identificáveis.
4. **Ausência de datas:** notas e faltas sem data impedem cortes semanais. A situação do
   discente tem data, e permitiria posicionar cancelamentos no tempo.
5. **Logs do AVA e entregas datadas:** não existem no portal aberto e, se existirem, só por
   acordo institucional.

## 4. O que pedir à UFRN (quando houver autorização)

Conforme `src/validation/data_contract.py`: campos obrigatórios (pseudônimo estável entre
tabelas, curso, componente, turma, período, datas de início das aulas, matrícula e
alteração de situação, situação final), campos opcionais em ordem de prioridade (**faltas
com data**, notas por unidade com data, entregas e prazos do AVA, logs do AVA) e atributos de
auditoria (sexo, deficiência, indicador socioeconômico), com base legal específica para os
dados sensíveis.
