# Revisão crítica do projeto SIACE — situação após as respostas do autor (2026-11-04)

Este arquivo substitui, como referência de estado, a revisão de 2026-11-03 (`REVISAO_CRITICA_PROJETO.md`, preservada). Os números citados estão em `RELATORIO_METRICAS.md` (seção "Revisão 2026-11-04"), com os caminhos dos runs.

---

## Como funciona a individualização por componentes conexas (método do sistema final)

**A contagem atual de árvores usa componentes conexas, não Watershed.** Isso vale para o teste de ponta a ponta (`runs/pipeline_test_cc_20261103_223209/`) e para a cidade (`runs/pipeline_cidade_cc_uniao_20261104/`, 11.077 árvores). O Watershed ficou na monografia como método testado e substituído.

Passo a passo, para cada tile:

1. **Máscara binária.** A U-Net ResNet-34 dá, para cada pixel, a probabilidade de ser copa. Com o limiar 0,65 (escolhido na validação), cada pixel vira 1 (copa) ou 0 (fundo).
2. **Limpeza morfológica.**
   - Fechamento com disco de raio 5 px: fecha falhas estreitas dentro da copa.
   - Remoção de buracos e de objetos menores que 32 px.
3. **Rotulagem.** Dois pixels de copa pertencem à mesma região se for possível ir de um ao outro andando só por pixels de copa vizinhos pela aresta (vizinhança-4: cima, baixo, esquerda e direita). O algoritmo (`scipy.ndimage.label`) dá um número inteiro diferente a cada região contínua.
4. **Filtro de área.** Regiões com menos de `a_min` = 3.000 px (≈ 3 m²) são descartadas como ruído. As demais são renumeradas a partir de 1.
5. **Uma região é uma árvore.** Cada região é recortada, classificada e georreferenciada.

Exemplo, onde `#` é copa e `.` é fundo:

```
. # # . . . # # .        . 1 1 . . . 2 2 .
# # # # . # # # #   ->   1 1 1 1 . 2 2 2 2      duas copas separadas: 2 árvores
. # # . . . # # .        . 1 1 . . . 2 2 .

. # # . . . # # .        . 1 1 . . . 1 1 .
# # # # # # # # #   ->   1 1 1 1 1 1 1 1 1      copas que se tocam: 1 árvore só
. # # . . . # # .        . 1 1 . . . 1 1 .
```

A limitação é justamente o segundo caso: o método não separa copas que se tocam. Por isso conta menos árvores do que existem (0,72 copa por árvore anotada no teste, IC 95% de 0,66 a 0,79). O Watershed tentava separar esse segundo caso. Nas máscaras urbanas, porém, ele errou mais do que acertou, em todas as variantes testadas (ver P1).

Código: `separate_tree_crowns_whatershed.py::label_connected_components`. Parâmetros congelados: `runs/watershed_vs_cc_val_20261103/frozen_instance_params.json`. Texto na monografia: `main_3.tex`, Seção `sec:metodo_watershed`.

---

## Situação de cada item

### Alta prioridade

**P1 — Watershed × componentes conexas. RESOLVIDO (as 3 alternativas).**
- **(a) Máscaras preditas na validação.** CC 0,657 contra Watershed 0,603 (`runs/watershed_vs_cc_val_20261103/`).
- **(b) Relato honesto.** O texto (resultados e conclusão) diz que o Watershed foi testado, por que falhou e que foi substituído.
- **(c) Marcadores h-maxima** (h de 0,10 a 1,00 m, `runs/watershed_hmax_predita_val_20261104/`):
  - o melhor resultado foi 0,633 (h = 1 m), abaixo das CC;
  - o F1 cresce com h e tende ao das CC, porque com h grande sobra um marcador por região;
  - conclusão: dividir as regiões da máscara predita sempre piorou.
- **Segmentação por instância treinada:** fica como trabalho futuro; não dá tempo até 07/11.
- **Monografia:** nova Tabela `tab:ws_hmax`.

**P2 — Validação da cidade. AMOSTRA SORTEADA; conferência pendente do autor.**
- Pasta: `runs/amostra_verificacao_cidade_20261104/`.
- **Arquivos:**
  - `amostra_100.csv`: planilha a preencher;
  - `amostra_100_pontos.geojson` e `amostra_100_copas.geojson`: para abrir no QGIS.
- **Desenho da amostra:**
  - 100 árvores do run final, fora dos 1.018 tiles anotados;
  - estratificada: 20 "sem identificação" + 16 de cada uma das outras 5 classes;
  - semente 42; coluna `peso_estrato` para estimar o total da cidade.
- **Colunas a preencher:**
  - `e_arvore` (s/n);
  - `classe_correta` (s/n/?);
  - `classe_observada`;
  - `copas_fundidas` (quantas árvores reais dentro da copa);
  - `deslocamento_m` (distância entre o ponto e o centro da copa na imagem de referência);
  - `referencia` (qual imagem foi usada);
  - `obs`.
- **Dica para medir o deslocamento:** use uma camada de satélite (por exemplo, Google ou Esri via XYZ Tiles) e a ferramenta de medir do QGIS. A ortofoto não serve de referência para isso, porque é a própria fonte das coordenadas.
- Depois de preenchida, eu calculo as precisões por estrato com IC e atualizo o texto. Há um `% PENDENTE` em `sec:georref`.

**P3 — Versionamento. PARCIAL.**
- O commit `fb38a59` guarda o estado da manhã de 04/11.
- Tudo o que foi feito depois (scripts, runs, LaTeX) ainda não está commitado.
- Sem push, como o autor pediu. Recomendo um novo commit quando ele quiser; não faço sem autorização.
- Os `.pth` continuam fora do git: vale uma cópia de `runs/` em disco externo.

**P4 — Incoerências e verbos no futuro. RESOLVIDO.**
- **Verbos no futuro:** a varredura não achou nenhum fora de comentários (as linhas com "será" restantes são comentários `%` do capítulo antigo "Resultados Esperados").
- **Corrigido:**
  - a classificação ainda dizia que os recortes vinham do "Watershed";
  - a U-Net multiclasse concluía com "segmentação binária e Watershed";
  - o número da união no teste era o do Watershed (360 → 273); agora é 301 → 225 com CC;
  - a introdução do capítulo de resultados;
  - a frase da introdução que afirmava resultado não medido ("possibilita a identificação eficiente de áreas carentes");
  - a legenda da figura de setorização;
  - um erro de concordância na Discussão;
  - a seção de métodos da segmentação e o resumo/abstract, que ainda falavam em "limiar padrão 0,5".

### Média prioridade

**P6 — Intervalos de confiança. RESOLVIDO.**
- **Método:** bootstrap por tile, pareado, B = 2000 (`bootstrap_ic.py` → `runs/bootstrap_ic_20261104/`). Descrito em `sec:metodo_avaliacao`; ICs nas tabelas e no texto.
- Nenhuma destas diferenças é distinguível do acaso:
  - semAug × principal;
  - as duas EfficientNet;
  - CC × Watershed no teste.
- A bananeira binária fica no limite (IC de −0,001 a 0,111).

**P7 — IoU como métrica e necessidade de refatorar os treinos. VERIFICADO; refatoração não necessária.**
- **Segmentação binária:** todos os scripts selecionam o checkpoint pelo IoU de validação, e agora todos têm o limiar escolhido pelo IoU de validação. Correto: IoU é a métrica padrão para máscaras.
- **Classificador de espécies (EfficientNet):** a seleção é por F1-macro de validação, e isso está certo. IoU é uma métrica de sobreposição de regiões. Para classificação de recortes, o equivalente por classe seria o índice de Jaccard, que é função crescente do F1 da mesma classe (J = F1 / (2 − F1)). Trocar F1-macro por Jaccard-macro quase não mudaria a escolha.
- **U-Net multiclasse:**
  - `resnet3.py` seleciona por mIoU (correto);
  - `resnet2.py` (variante CE) seleciona por perda de validação, além de usar outro otimizador e outra normalização.
- **Decisão:** em vez de retreinar, os confundidores foram declarados na Seção `sec:unet_mc`. Se o autor quiser a comparação limpa, basta alinhar o `resnet2.py` ao `resnet3.py` e retreinar (treino é rodado pelo autor).

**P8 — Recalibrar o Watershed nas máscaras preditas. RESOLVIDO.**
- Grade completa: janela 41–251 × área 1000–3000.
- Melhor resultado: f = 251, a_min = 3000, F1 0,612 (antes 0,603). Continua abaixo das CC (0,657).

**P9 — União na borda. RESOLVIDO (decisão minha).**
- **Problema:** a regra antiga (união encadeada) juntava, através do tile vizinho, copas que as CC já tinham separado no mesmo tile. 292 árvores tinham 2 ou mais partes do mesmo tile; um grupo tinha 803 partes.
- **Regra nova** (`uma_parte_por_tile`, padrão em `tree_georef.merge_border_crowns`):
  - pares unidos em ordem decrescente de contato;
  - no máximo uma parte por tile em cada árvore.
- **Reprocessamento:** a cidade foi re-unida a partir dos parciais salvos, sem GPU: 11.077 árvores (antes 10.251).
- **Testes:**
  - a regra antiga continua disponível (`--merge-rule encadeada`) e reproduz exatamente o resultado antigo no teste;
  - 2 testes unitários novos.
- **Limite que continua:** maciços de vegetação contínua. 175 objetos com mais de 500 m² somam 61% da área de copa, e o maior tem cerca de 9 ha. Isso está declarado no texto.

**P10 — Classe 1. RESOLVIDO no texto.** Ficou explicado que a classe 1 é a escolha do anotador quando a espécie não era reconhecível com segurança na imagem aérea. Por isso é heterogênea, e isso limita o classificador (`sec:dados_cls`).

**P11 — Limiar travado em 0,5 e nome "P1". RESOLVIDO.**
- **Limiares:**
  - as 5 configurações herdadas tiveram o limiar escolhido na validação (`reavaliar_limiar_herdados.py` → `runs/limiar_val_herdados_20261104/`);
  - a reimplementação reproduziu o limiar 0,65 da principal;
  - o run `20261101_185936_512semAug2` foi excluído: o treino foi interrompido (há o run completo `treino_…`).
- **Resultado:** a 512semAug empatou com a principal na validação (0,728 × 0,726, IC da diferença contém zero). A principal foi mantida, e a decisão está justificada no texto.
- **Nome:** na monografia, "run principal" e "P1" viraram "U-Net ResNet-34" (o nome real: U-Net com codificador ResNet-34). O diretório `runs/p1_20261101_140519/` **não** foi renomeado, por causa da restrição original de não mover/renomear runs. Ele só aparece como caminho nas fontes das tabelas.

**P12 — Bananeira contra as outras. RESOLVIDO.**
- O run do autor (`classif_20261103_115814_efficientnet_binaria_bananeira`) foi comparado: F1 0,915 × 0,863; falsos positivos 10 → 3.
- AUC igual nos dois (0,964 × 0,963): o ganho está no ponto de decisão, não na separação entre as classes.
- Nova subseção `sec:cls_binario` e Tabela `tab:cls_binario`.

### Itens não tocados nesta rodada
- **P5 — informações do autor:** modelo do drone e da câmera, data, sobreposição, software do ortomosaico, critério dos 1.018 tiles. Os `% PENDENTE` continuam no texto.
- **P13–P17 (baixa prioridade):** sem mudança.

---

## Pendências objetivas para a entrega (07/11)
1. Conferir as 100 árvores (P2) e me devolver o CSV preenchido.
2. Informações de P5.
3. Mapa `mapa_resultado_especies.png` no QGIS, a partir de `runs/pipeline_cidade_cc_uniao_20261104/arvores.geojson` (não usar os runs anteriores da cidade).
4. Compilar no Overleaf e verificar warnings (não há LaTeX local).
5. Novo commit (com autorização).
6. Opcional: descrever o que difere entre os scripts do benchmark (`% PENDENTE` em `sec:res_aug`).

## Avaliação final
- O protocolo está coerente de ponta a ponta. Toda escolha (checkpoint, limiar, método de individualização, área mínima, variante do classificador) foi feita na validação, e o teste foi usado uma vez por método.
- O texto não atribui ganhos a diferenças que o bootstrap não sustenta.
- **Limitações principais, todas declaradas:**
  - subcontagem por copas que se tocam;
  - maciços contados como uma árvore;
  - classe 1 heterogênea;
  - classes raras com poucos exemplos;
  - ausência de validação em campo.
