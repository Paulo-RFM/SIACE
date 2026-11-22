# Revisão crítica do projeto SIACE (2026-11-03)

Escopo: código em `lab_2026/`, runs em `lab_2026/runs/`, monografia (`main_3.tex`, `resultados.tex`).
Contexto: entrega ao orientador em 07/11/2026; banca em 05/12/2026.
Todos os números citados vêm de arquivos existentes; a fonte está indicada.

---

## 0. GNSS: o que existe e o que foi corrigido

**Não existe módulo GNSS no código.** Nenhum script lê GNSS, EXIF, RTK ou log de voo (busca por `gnss|gps|rtk|exif`
em todos os `.py`). O "GNSS" do projeto é:

1. a subseção GNSS da monografia;
2. indiretamente, o georreferenciamento das árvores (`tree_georef.py`), que lê as coordenadas gravadas nos
   GeoTIFF. Essas coordenadas vêm da ortofoto, que por sua vez vem do GNSS embarcado no VANT.

### Problemas encontrados e corrigidos

| # | Problema | Tipo | Correção |
|---|---|---|---|
| G1 | O texto dizia que o GNSS "será usado para georreferenciar as imagens e recolher a localização exata das árvores". Nenhuma árvore foi medida com GNSS; a posição vem dos metadados da ortofoto. | Inconsistência conceitual (e verbo no futuro) | Subseção reescrita: o GNSS do VANT alimenta a ortofoto; a árvore é posicionada pela Eq. de georreferenciamento; exatidão absoluta não medida. Novo `\label{sec:gnss}` |
| G2 | "Alta exatidão" afirmada sem evidência. Não há registro de pontos de controle (GCP) nem RTK/PPK. | Afirmação sem suporte | Texto distingue exatidão relativa (pixel de ~3 cm) de absoluta (não medida). `% PENDENTE` pede modelo do VANT e RTK |
| G3 | `resultados.tex` afirmava como fato que "não foram usados pontos de controle". | Afirmação não verificada | Trocado por "não há registro de ...", com referência à Seção GNSS |
| G4 | `read_geotransform` aceitava silenciosamente GeoTIFF com rotação (tag 34264), PixelIsPoint (GeoKey 1025=2), unidade angular diferente de grau ou escala inválida. A fórmula daria coordenadas erradas sem aviso. | Falha de robustez | Agora falha explicitamente nesses casos. Verificado: 1.018 tiles anotados + 2.000 tiles da cidade passam (0 falhas) |
| G5 | Georreferenciamento sem testes automatizados. | Manutenibilidade | `tests/test_georef.py` (5 testes: canto, sentido do eixo y, meio pixel no centróide, continuidade da grade entre tiles vizinhos, tamanho do pixel ≈ 3 cm). 35/35 testes passam |

Conferido e **correto** (não precisou mudar):
- tiles com `GTRasterTypeGeoKey = 1` (PixelIsArea), então o tiepoint é o canto do pixel e o centróide usa +0,5 px;
- contorno em bordas de pixel;
- grade contínua entre tiles (o vizinho a leste começa em `lon0 + 512·sx`);
- pixel de cerca de 3,19 cm × 3,19 cm (sx = 3,04e-7°, sy = 2,89e-7° a −19,2° de latitude);
- áreas em UTM 23S (EPSG:31983), adequado a Rio Paranaíba.

### Causa provável da confusão

A subseção GNSS foi escrita na fase de proposta (TCC1), quando se previa coleta em campo e imagens de rua. A
implementação real seguiu outro caminho (georreferenciamento herdado da ortofoto) e o texto não foi atualizado.
O mesmo padrão aparece em outras seções (ver P4).

---

## 1. Problemas encontrados

### Alta prioridade

**P1. O Watershed calibrado não supera rotular manchas conectadas.**
Novo run `runs/watershed_vs_cc_val_20261103/`, somente na validação, com máscaras de referência:

| Método | F1 |
|---|---|
| Componentes conexas (mesma limpeza, `min_area` = 1000) | **0,880** |
| Watershed calibrado | 0,832 |

- O `peak_footprint` escolhido é de 251 px, quase meio tile (cerca de 7,5 m). Com ele, o Watershed quase não divide copas: em 134 dos 203 tiles dá o mesmo número de instâncias que as componentes conexas.
- Quando divide, erra na maioria das vezes: 59 instâncias a mais para só 7 acertos a mais.
- A tendência do `val_sweep.csv` (F1 sobe à medida que a janela aumenta) já indicava isso.

*Risco:* a banca pode perguntar por que usar Watershed, e o texto atual atribui a ele o ganho de 0,607 para 0,859. Esse ganho veio de **parar de dividir demais**, não de separar bem copas que se tocam.

*Alternativas:*
- (a) Repetir a comparação com as máscaras **preditas** pelo P1 na validação. É o cenário real: na máscara predita as copas se fundem mais, e o Watershed pode ajudar. Custa cerca de 1 min de GPU para gerar as predições de validação.
- (b) Se as componentes conexas empatarem ou vencerem, reportar isso honestamente: o Watershed com marcadores por máximos da distância não separou bem as copas urbanas, e a separação de instâncias fica como trabalho futuro.
- (c) Testar marcadores mais robustos (h-maxima da transformada de distância com altura mínima em metros) ou segmentação por instância treinada.

**P2. Os resultados da cidade não têm nenhuma validação direta.**
- As 14.440 árvores e a distribuição por classe são extrapolação de um modelo avaliado em 204 tiles.
- A precisão de "mangueira" no teste de ponta a ponta foi de 0,34, então as 876 mangueiras estão provavelmente superestimadas.
- *Alternativa barata e forte para a banca:* sortear cerca de 100 árvores da cidade (estratificadas por classe) e conferir visualmente no QGIS se é árvore e se a classe é plausível. Isso dá uma precisão medida na cidade, com intervalo de confiança.

**P3. Nada do trabalho atual está versionado nem tem backup.**
- O git tem 16 commits e 137 arquivos alterados ou não rastreados (todos os scripts novos, relatórios e runs).
- Os `.pth` estão no `.gitignore` e `runs/` ocupa 4,8 GB num único disco.
- Uma falha de disco perde os checkpoints (incluindo o P1) e todo o trabalho da auditoria.

**P4. A seção "Avaliação dos Resultados" (`main_3.tex`) promete o que não foi feito, no futuro.**
- Promete comparação com "outras fontes de dados" e robustez em "áreas urbanas e rurais".
- Usa "será calculada", "será analisado" etc.
- O mesmo vale para a subseção VANT ("serão essenciais", "precisão georreferenciada") e a introdução de "Ferramentas e Softwares" ("serão adotados").
- Numa monografia final, isso é inconsistência direta com os Resultados.

**P5. Informações do autor ausentes que a banca vai perguntar:**
- modelo do VANT, câmera, data do voo e sobreposição;
- software de ortorretificação;
- critério de seleção dos 1.018 tiles;
- quem anotou e com qual critério de espécie.

Há `% PENDENTE` no texto para cada um.

### Média prioridade

**P6. Nenhuma métrica de teste tem intervalo de confiança.**
- Seed única, e classes raras no teste: árvore seca com 3 recortes, jabuticabeira com 6.
- Diferenças como 0,639 × 0,603 (as duas EfficientNet) podem não ser significativas.
- *Alternativa sem retreinar:* bootstrap por tile sobre as predições já salvas (`test_predictions.csv`, `arvores_casadas.csv`).

**P7. A comparação das U-Nets multiclasse tem confundidores.**
- `unet_mc_ce` (resnet2) usa `normalize: none`, Adam e seleção por `val_loss`.
- As variantes CE+Dice (resnet3) usam normalização ImageNet, AdamW e seleção por IoU.
- O texto atribui a diferença à perda, mas quatro fatores mudam juntos.
- *Correção mínima:* declarar isso na Seção `sec:unet_mc`.

**P8. O Watershed foi calibrado em máscaras de referência e aplicado em máscaras preditas.**
- As distribuições são diferentes: a máscara predita é mais "borrada" e funde mais copas.
- O F1 caiu de 0,859 para 0,626 com a máscara predita.
- A calibração deveria ser feita na máscara predita da validação (ligado a P1).

**P9. A união de copas na borda entre tiles pode fundir árvores distintas.**
- O critério é tocar com tolerância de 10 cm.
- Duas árvores vizinhas cortadas pela mesma borda viram uma (4.472 uniões na cidade).
- Contribui para a subcontagem e para a área média alta da mangueira (226 m²).
- *Alternativa:* exigir que as partes se complementem na borda (sobreposição do segmento de borda maior que X%).

**P10. A classe 1, "árvore sem identificação", não é uma espécie.**
- É um "resto" que mistura espécies. Parte da confusão com mangueira pode ser de mangueiras reais anotadas como classe 1.
- Isso limita o teto de qualquer classificador.
- Deve ser explicitado como limitação e alvo de revisão das anotações.

**P11. As configurações do benchmark de segmentação não são comparáveis entre si.**
- Só o P1 teve limiar escolhido na validação; as outras usam 0,5.
- Já está declarado no texto. Falta reavaliá-las ou descrever as variantes (`% PENDENTE` D5).

**P12. O experimento binário (bananeira contra as outras) está implementado, mas não foi rodado.**

### Baixa prioridade

**P13.** O ambiente `siace` tem o GDAL quebrado (falta `libnetcdf.so.19`): sem rasterio, fiona ou GeoPackage. Por isso a saída é GeoJSON, e o `copas.geojson` tem 418 MB, pesado no QGIS. Não há `environment.yml` nem `requirements.txt`.

**P14.** A pasta do LaTeX tem `main.tex` e `main2.tex` antigos, com labels que duplicam os do `main_3.tex`. Não afeta a compilação do `main_3.tex`, mas confunde.

**P15.** `labels2/OrtoRP-3cm_082_013 - Copia.json` continua no disco. Está excluído do manifesto, mas outros scripts que fazem glob em `labels2/` (`make_mask.py`) podem lê-lo.

**P16.** Não há compilação LaTeX local. Toda checagem foi estática (refs, labels, ambientes). Só o Overleaf mostra warnings de overfull, figuras flutuando ou citações faltando no `.bib`.

**P17.** As 53 adjacências diagonais entre conjuntos do split são uma limitação residual. Já está declarada; aceitável.

---

## 2. Melhorias recomendadas

| Área | Recomendação |
|---|---|
| **Arquitetura** | Separar o pipeline em estágios com artefatos persistidos (máscaras → instâncias → classificação → geo), cada um com proveniência. Hoje é monolítico, mas retomável; está bom para o TCC. Para separar instâncias, avaliar segmentação por instância treinada em vez de Watershed |
| **Organização do código** | Um `README` de "como reproduzir" com a sequência de comandos e o run gerado por cada um. Scripts históricos (`resnet2/3`, variantes `512*`) movidos para `legacy/` (sem apagar) |
| **Organização do documento** | Remover do texto final o que é proposta não executada (seções VANT, Avaliação, Resultados Esperados comentado). Mover o fluxograma para o início da Metodologia e citá-lo no texto |
| **Desempenho** | A cidade leva cerca de 93 min. O gargalo é o Watershed na CPU (4 núcleos). Se P1 confirmar as componentes conexas, o pós-processamento fica várias vezes mais rápido. Para outras cidades, processar por blocos maiores com sobreposição eliminaria a união por borda |
| **Segurança** | Para o OSM: token OAuth2 só em variável de ambiente, nunca no código nem no git. Testar só no servidor de desenvolvimento. Os `.pkl` em `partes/` usam `pickle` (carregar só arquivos próprios). Os runs gravam caminhos absolutos com o nome de usuário; anonimizar se forem publicados |
| **Escalabilidade** | GeoPackage ou FlatGeobuf em vez de GeoJSON (exige consertar o GDAL). Índice espacial já existe (STRtree) |
| **Manutenibilidade** | Testes cobrem a divisão, o manifesto e o georreferenciamento (35). Faltam testes do `pipeline_cidade.py` (tile de borda, tile transparente: os dois bugs desta semana) e do avaliador de detecção |
| **Documentação** | Exportar o ambiente (`conda env export > environment.yml`). Consolidar `RELATORIO_METRICAS.md` como fonte única dos números da monografia (já está quase) |

---

## 3. Informações ausentes

- **Dados de aquisição:**
  - modelo do VANT e da câmera;
  - data e hora do voo;
  - altura (200 m, confirmada pelo autor; verificar coerência com pixel de 3 cm);
  - sobreposição;
  - RTK/PPK e uso de GCP.
- **Processamento da ortofoto:** software de ortorretificação e ferramenta de recorte dos tiles.
- **Critérios:** seleção dos 1.018 tiles; regras de anotação (o que conta como uma copa; quando usar a classe 1).
- **Métricas:**
  - intervalos de confiança (P6);
  - precisão medida na cidade (P2);
  - erro absoluto de posição (G2);
  - comparação Watershed × componentes conexas na máscara predita (P1).
- **Diagramas:** o mapa da cidade (`mapa_resultado_especies.png`, a fazer no QGIS). Talvez um exemplo de erro típico (copas fundidas, mangueira falsa).
- **Referências:**
  - pedido de verificação das entradas bib marcadas como "verificar" (Crossref falhou);
  - referência para a exatidão típica do GNSS sem RTK, se o texto citar ordem de grandeza;
  - referência do Watershed/transformada de distância.
- **Definições:** "árvore" para a anotação (arbustos? cercas vivas?) e a unidade de contagem quando copas se fundem.

---

## 4. Pendências (objetivas)

| # | Pendência | Tipo | Quem |
|---|---|---|---|
| 1 | Commit + backup externo de `runs/` (pelo menos P1, EfficientNet selecionada, cidade) | Proteger | Autor |
| 2 | Comparar Watershed × componentes conexas na máscara predita (validação) e decidir o texto | Validar | Claude (com GPU livre) |
| 3 | Reescrever "Avaliação dos Resultados", VANT e introdução de Ferramentas no passado, só com o que foi feito | Corrigir | Claude |
| 4 | Mapa da cidade no QGIS | Documentar | Autor |
| 5 | Dados do VANT/voo/ortofoto/seleção de tiles (preencher os `% PENDENTE`) | Documentar | Autor |
| 6 | Validação visual por amostragem na cidade (~100 árvores) | Validar | Autor (roteiro e sorteio: Claude) |
| 7 | Intervalos de confiança por bootstrap | Implementar | Claude |
| 8 | Experimento binário (bananeira; opcional mangueira) e subseção | Executar/documentar | Autor roda, Claude escreve |
| 9 | Nota dos confundidores das U-Nets multiclasse | Corrigir | Claude |
| 10 | Variantes do benchmark de segmentação (D5): descrever ou reduzir a tabela | Documentar | Autor + Claude |
| 11 | Compilar no Overleaf e revisar warnings/figuras | Validar | Autor |
| 12 | Exportar `environment.yml` | Documentar | Claude |
| 13 | Envio ao OSM (depois de 07/11) | Implementar | Extra |

---

## 5. Plano de ação (priorizado)

| Prioridade | O que fazer | Motivo | Impacto esperado | Quando |
|---|---|---|---|---|
| **Alta** | Commit + cópia de `runs/` para HD externo ou nuvem | Perda total possível | Elimina o maior risco não técnico | Hoje |
| **Alta** | Watershed × componentes conexas com máscara predita na validação; congelar o melhor; reavaliar o teste uma vez **se** mudar o método (declarando) | Banca vai questionar o Watershed | Texto honesto; talvez F1 de ponta a ponta maior e pipeline mais rápido | 03–04/11 |
| **Alta** | Reescrever as seções em futuro / não executadas (P4) | Inconsistência visível na primeira leitura | Monografia coerente | 04/11 |
| **Alta** | Preencher dados de aquisição e critérios (P5) | Perguntas certas da banca | Metodologia completa | 04–05/11 (autor) |
| **Alta** | Mapa da cidade | A figura é o "produto" do TCC | Fecha a seção da cidade | 05/11 (autor) |
| **Média** | Validação visual de ~100 árvores da cidade | Os números da cidade não têm validação | Transforma estimativa em resultado medido | 05–06/11 |
| **Média** | Bootstrap (IC 95%) para as métricas principais | Classes raras, seed única | Comparações defensáveis | 05/11 |
| **Média** | Experimento binário | Pedido do autor; responde "um modelo por espécie ajuda?" | Uma subseção a mais | Quando a GPU estiver livre |
| **Média** | Confundidores U-Net MC; classe 1 como limitação | Rigor | Evita crítica metodológica | 05/11 |
| **Baixa** | `environment.yml`, `README` de reprodução, mover scripts legados | Reprodutibilidade | Facilita arguição e continuidade | 06/11 |
| **Baixa** | GDAL / GeoPackage; testes do pipeline | Escala e manutenção | Arquivos menores, menos regressões | Depois de 07/11 |
| **Extra** | OSM | Contribuição social | Ver abaixo | Depois de 07/11 |

**Atenção sobre o OSM.** As regras da comunidade OSM para importações e edições automatizadas exigem discussão
prévia e documentação antes de enviar dados gerados em massa. Com precisão de 0,34 para mangueira, um envio
automático das 14.440 árvores seria inadequado. Alternativa responsável:
- enviar só `natural=tree` (sem espécie) nas detecções de alta confiança; ou
- usar revisão humana (ex.: tarefas no MapRoulette);
- testar sempre antes em `master.apis.dev.openstreetmap.org`.

---

## 6. Avaliação final

**Pontos fortes**
- Protocolo de avaliação rigoroso para um TCC:
  - divisão espacial persistida com hash;
  - seleção na validação e teste avaliado uma única vez;
  - proveniência completa por run;
  - auditoria que encontrou e descartou um classificador com vazamento.
- Sistema de ponta a ponta funcionando, da imagem ao GeoJSON, aplicado a 61.426 tiles. Georreferenciamento verificado visualmente e por testes.
- Resultados honestos e bem qualificados: o texto distingue referência × predita, isolado × ponta a ponta.
- Achados técnicos com valor de discussão: a acurácia global engana; o bug de paleta das U-Nets multiclasse; o efeito da calibração.

**Principais riscos**
1. O papel do Watershed (P1): é o ponto mais vulnerável na arguição hoje.
2. Números da cidade sem validação (P2).
3. Perda de dados por falta de backup (P3).
4. Texto residual da proposta (P4, GNSS, VANT): passa impressão de trabalho inacabado.
5. Classes raras com 3–6 amostras de teste: métricas por classe instáveis (já declarado).

**Grau de maturidade atual:** **bom, cerca de 80% do necessário para a entrega de 07/11.**
- O núcleo técnico (segmentação, classificação, pipeline, georreferenciamento) está pronto e avaliado.
- O que falta é principalmente consistência do texto, informações do autor, o mapa e uma decisão sobre o Watershed.
- Nada disso exige novo treinamento.

**Para estar pronto para entrega/apresentação:**
1. Resolver P1 (decisão + texto).
2. Eliminar texto de proposta (P4).
3. Preencher P5 e o mapa.
4. Backup.
5. Compilar no Overleaf sem erros.

Validação por amostragem na cidade e ICs não são obrigatórios para 07/11, mas são os itens que mais fortalecem
a defesa de 05/12.

---
## Atualização 2026-11-03 (tarde)
- P1 verificado com máscara PREDITA na validação: componentes conexas F1 0,641 (min_area 1000) / 0,656 (2000)
  contra 0,603 do watershed congelado. Texto atualizado (`resultados.tex`, nova subseção `sec:ws_vs_cc`; conclusão).
  O pipeline ainda usa o watershed; trocar exige reavaliar o teste uma vez (declarado) e rodar a cidade de novo.
- P4 resolvido: "Avaliação dos Resultados" reescrita no presente/passado só com o que foi feito (com Eq. de
  precisão/recall/F1 e casamento húngaro); VANT, introdução de Ferramentas e uma frase do referencial corrigidas.

## Atualização 2026-11-04 (madrugada)

- Cidade reprocessada com componentes conexas (`runs/pipeline_cidade_cc_20261103/`): 10.251 árvores. Seção `sec:aplicacao_cidade`, resumo/abstract e conclusão atualizados. Pendência de coerência entre método do sistema e números da cidade: **resolvida**.
- O mapa (`mapa_resultado_especies.png`) deve ser gerado a partir de `runs/pipeline_cidade_cc_20261103/arvores.geojson`.
