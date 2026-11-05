# Governança, Privacidade e Uso Responsável

> Fase 05 · 2026-11-05. Este documento orienta o projeto; **não** é parecer jurídico. As
> decisões de base legal cabem ao controlador dos dados (a instituição) e ao seu encarregado (DPO).

## 1. Princípios do EduGuard-IA (preservados)

1. O EduGuard-IA continua **independente** do aplicativo Flutter. A comunicação futura será por
   **API REST**, que **ainda não foi autorizada** (Fase 06 ou posterior).
2. O modelo **não toma decisões acadêmicas**. Ele produz uma **estimativa de risco** para apoiar
   uma conversa pedagógica feita por pessoas.
3. As previsões são apresentadas como **probabilidades com incerteza**, nunca como certezas ou
   rótulos permanentes.
4. Thresholds e critérios de acompanhamento são definidos **com a equipe pedagógica**
   (`docs/THRESHOLD_METHODOLOGY.md`).
5. Gênero, deficiência e indicador socioeconômico **não são preditores** (Q16/Q17). São usados
   só para auditoria de equidade.

## 2. LGPD — preparação para dados brasileiros

| Tema | Diretriz |
|---|---|
| Base legal | A definir pelo controlador. Para pesquisa, a LGPD prevê a realização de estudos por órgão de pesquisa, com anonimização sempre que possível (art. 7º, IV; art. 11, II, "c" para dados sensíveis) |
| Dados sensíveis | Deficiência (dado de saúde) é dado pessoal sensível (art. 5º, II). Só para auditoria, com base legal específica |
| Minimização | Só os campos do contrato de dados; faixa etária em vez de data de nascimento; nada de nome, CPF, e-mail ou matrícula real |
| Pseudonimização | Feita pela instituição antes da entrega (hash com segredo que não sai da instituição) |
| Finalidade | Validação científica. Proibido reaproveitar para outros fins sem nova autorização |
| Segurança | Ambiente com acesso restrito e registro de acesso; dados fora do Git (`.gitignore`); exclusão ao fim do projeto conforme o acordo |
| Transparência | Estudantes informados sobre a existência do sistema e seu caráter de apoio, quando houver uso real |
| Avaliação de impacto | Elaborar relatório de impacto (RIPD) antes de qualquer uso operacional |
| Reidentificação | Proibida qualquer tentativa |

## 3. Acompanhamento humano que não depende só do score (Q18)

O score é **um** sinal. Para não deixar de fora estudantes cujo risco o modelo subestima (como
os estudantes com deficiência, ver `docs/FAIRNESS_AUDIT.md`):

1. **Revisão humana obrigatória** de qualquer alerta antes de contato. Sem ações automáticas.
2. **Canais independentes do score:** núcleos de acessibilidade e apoio estudantil mantêm o
   acompanhamento próprio dos estudantes atendidos, com ou sem alerta do modelo.
3. **Gatilhos comportamentais explícitos e transparentes** (ex.: nenhuma entrega de uma atividade
   vencida), revisados pela equipe, além da probabilidade.
4. **Busca ativa por demanda espontânea:** o estudante pode pedir apoio a qualquer momento.
5. **Monitoramento por grupo:** a cada período, comparar a taxa de alerta e a taxa observada de
   cancelamento por grupo. Se houver subestimação persistente, rever o modelo, **não** criar threshold por grupo
   (decisão atual: sem thresholds específicos por grupo).
6. **Registro e retorno:** documentar as intervenções e seus resultados para avaliar efetividade
   (insumo para o Q19).

## 4. O que fica proibido até nova autorização

API, integração com o Flutter, previsões operacionais, rotulagem de estudantes reais, uso de
dados brasileiros sem autorização e qualquer decisão automatizada com efeito sobre o estudante.
