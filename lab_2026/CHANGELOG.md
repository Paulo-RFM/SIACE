# CHANGELOG

## 2026-11-22 — requirements.txt, inventário e scripts de João Pessoa

- `requirements.txt` do projeto (versões do ambiente conda siace; GDAL pelo ambiente geo).
- `FERRAMENTAS_E_SCRIPTS.txt`: hardware, bibliotecas, ferramentas e todos os scripts por etapa.
- `joao_pessoa/`: cópias dos scripts usados em ~/proj_JoaoPessoa211126 (preparo, mapas, cobertura, conferência).

## 2026-11-22 — Contêiner Docker (inferência)

- Nova pasta `docker/`: `Dockerfile` (base GDAL 3.6.2 + PyTorch 2.5.1 cu121), `requirements.txt`, `siace.sh` (entrada: `processar` ortomosaico → tiles → pipeline; `--modelo 3cm|8cm`; zona UTM automática), `preparar_ortomosaico.py` (generaliza `preparar_jp_3cm.py`), `slicer.py` (cópia de `~/proj_JoaoPessoa211126/slicer.py`), `baixar_modelos.sh` (pesos por release do GitHub, conferidos por SHA-256) e `LEIAME.md`.
- `.dockerignore` em lab_2026 com lista de permissão (só código do pipeline, modelos e parâmetros congelados).
- Código do sistema não alterado. Fluxo testado fora do Docker (mesma estrutura de /app, recorte de João Pessoa); a imagem Docker não foi construída (Docker ausente nesta máquina).

## 2026-11-22 — Monografia: origem da fusão de copas

- `resultados.tex` (seção da aplicação ao ortomosaico, após os maciços de vegetação): nova frase explicando que a fusão de copas não decorre do treino (máscaras binárias 0/255, verificado em `/home/paulomaciel/proj/masks`), e sim da individualização e da união entre tiles.
- Backup: `audit_backups_20261102/latex_antes_origem_fusao_20261122/`.

## 2026-11-21 — Experimento de resolução (8 cm e 15 cm) e João Pessoa

- **Concluído em 22/11, às 6h13.** IoU de teste de cada modelo na sua própria resolução: 3 cm, 0,773 (IC 0,739–0,805); 8 cm, 0,772 (0,740–0,802); 15 cm, 0,750 (0,720–0,779).
- O modelo atual testado em 8 cm tem 0,772 e, em 15 cm, 0,733. Diferenças pareadas contra o atual:
  - 8 cm no teste de 15 cm: +0,037 (IC +0,017 a +0,054);
  - 15 cm no teste de 15 cm: +0,016 (IC −0,005 a +0,035).
- João Pessoa (14.554 tiles), árvores por modelo:
  - atual: 5.110 árvores, 43,8 ha de copa;
  - 8 cm: 5.773 árvores, 46,4 ha;
  - 15 cm: 5.217 árvores, 52,6 ha;
  - sem referência para medir o acerto.
- Gráficos refeitos depois de ajustar o espaçamento do `exemplo_predicoes.png`.

- `degradar_resolucao.py` (novo): simula 8 cm e 15 cm a partir dos 1.018 tiles anotados. A média de área reduz cada tile a 204 px e a 109 px (pixel efetivo de 0,0801 e 0,1498 m), e a interpolação bilinear devolve a imagem a 512 px na grade de 3,19 cm. Máscaras, divisão espacial, georreferência e pixels sem dado ficam iguais. Os conjuntos estão em `/home/paulomaciel/proj/resolucao_20261121/imagens_{8,15}cm`.
- `runs/resolucao_20261121.sh` (nohup), em cadeia:
  - treina `res_segformer_8cm_20261121` e `res_segformer_15cm_20261121`, com o mesmo script e semente da SegFormer do sistema;
  - avalia 3 modelos × 3 resoluções de teste (a do modelo atual em 3 cm é reaproveitada);
  - roda `comparar_resolucao.py`;
  - aplica os 3 modelos a João Pessoa.
  - As saídas ficam em `runs/resolucao_comparacao_20261121/`.
- `comparar_resolucao.py` (novo): métricas, IC por bootstrap, diferença pareada contra o modelo atual, curvas de treino e de limiar, matriz de IoU, exemplo de predições; com `--joao-pessoa`, contagens e recorte.
- `tree_georef.py`: a variável de ambiente opcional `SIACE_UTM_EPSG` permite outra zona UTM (31985 em João Pessoa). Sem ela, nada muda. Backup em `audit_backups_20261102/tree_georef_antes_utm_env_20261121.py`.
- João Pessoa (`~/proj_JoaoPessoa211126`):
  - `slicer.py` ganhou `--pular-brancos` (backup em `slicer_v1_20261121.py`);
  - `preparar_jp_3cm.py` reprojeta para EPSG:4326 com pixel de 3,19 cm (bilinear) e gera 14.554 tiles; 16.406 tiles totalmente brancos foram pulados.

## 2026-11-21 — Contagem no perímetro urbano

- As 13.491 árvores cobrem o ortomosaico inteiro, incluindo áreas rurais; só 4.696 (35%) têm o centro no perímetro (`Placemark.kml`).
- Novo run `runs/pipeline_perimetro_segformer_20261121/`: sistema congelado (SegFormer MiT-B1 + h-maxima + EfficientNet-B0 semente 42) nos 13.360 tiles que tocam o perímetro (12.969 de tiles512 + 391 de novas_imagens, mesma lista da cobertura), ~28 min. Resultado: **4.854 árvores no perímetro** (`contagem_perimetro.json`).
- Conferência das 100: 38 no perímetro (31 árvores reais, 21 com classe plausível); fora, 37/55 e 19/57 (dúvidas todas fora).
- `resultados.tex`: "área urbana" → "ortomosaico"/"área mapeada" onde o número é do ortomosaico; parágrafo com a contagem do perímetro; recorte descritivo da conferência. `main.tex`: resumos PT/EN, Métodos e Conclusão. Backup: `audit_backups_20261102/latex_antes_perimetro_20261121/`.

## 2026-11-21 — Apresentação reorganizada em 4 partes

- `apresentacao_defesa/gerar_apresentacao.py` (fora do repositório): 20 → 28 slides, em quatro perguntas: 1. O que é? (objetivo, dados, sistema, individualização); 2. O que resolve? (problema, protocolo, segmentação, classificação, sistema no teste, conferência de 100 árvores, limitações); 3. Quais benefícios? (novo slide de benefícios com Roy et al., 2012; cidade, detalhe, cobertura); 4. Extras (sementes da segmentação, escolha da individualização, matriz de confusão, sementes da EfficientNet, OSM). Slides novos: roteiro, 4 divisórias, conferência na cidade, benefícios e sementes da EfficientNet. "Ilhas de calor" removido (sem fonte). Limitações, cidade e conclusão citam a conferência (73%). Backup: `audit_backups_20261102/gerar_apresentacao_antes_4partes_20261121.py` e `defesa_antes_4partes_20261121.pptx`.

## 2026-11-21 — Revisão final da monografia (furos e referências)

- `main.tex`:
  - Introdução: benefícios das árvores urbanas agora com referência (Roy, Byrne e Pickering, 2012; resumo conferido no PDF de acesso aberto) e reescritos para o que o artigo sustenta.
  - Referencial: removidos trechos sem fonte (telemetria e estações terrestres; faixa "50 cm a 1 km" de resolução, desatualizada; câmera RGB e mapas topográficos; parágrafo satélite × drone); caducifólias vinculadas a neto2012; primeira pessoa trocada por impessoal; erros de português.
  - Discussão: "georreferenciamento preciso" substituído por remissão à exatidão absoluta (Seção GNSS).
  - Métodos: chamadas no texto para `fig:setorização` e `fig:Pipeline`; citações de Watershed (Vincent e Soille, 1991), h-maxima (Soille, 2004), algoritmo húngaro (Kuhn, 1955), IoU ≥ 0,5 (Everingham et al., 2010), kappa (Cohen, 1960), bootstrap (Efron e Tibshirani, 1993), ImageNet (Deng et al., 2009) e software (PyTorch, smp, scikit-learn, scikit-image, SciPy, QGIS, GDAL, LabelMe).
  - Conclusão: média das 4 sementes da EfficientNet e limitação da seleção; avaliador único na conferência.
- `resultados.tex`: parágrafo "Variação entre treinos" e tabela `tab:cls_sementes` (runs/classif_seeds_efficientnet_20261120, regra 3 do critério: escolha não se confirma, modelo mantido); Wilson, Dice (Milletari et al., 2016) e AdamW (Loshchilov e Hutter, 2019) citados; conferência "como único avaliador e sem verificação em campo".
- `referencias.bib`: 20 entradas novas, DOIs conferidos na Crossref.
- Backup: `audit_backups_20261102/latex_antes_revisao_final_20261121/`.

## 2026-11-21 — Resultado da conferência das 100 árvores

- O `amostra_100_preenchida.csv` chegou idêntico ao modelo (vazio); as respostas vieram da tabela de atributos do QGIS colada na conversa e foram transcritas em `runs/conferencia_100_arvores_20261121/respostas_transcritas_qgis.csv` (e_arvore, classe_correta, copas_fundidas; números 1/3 e "0" nas linhas 1–4 e 53 guardados como `_bruta`, sem uso).
- `analisar_conferencia.py` → `metricas_por_classe.csv`, `metricas_conferencia.json`: precisão de detecção na cidade (ponderada por estrato) 73,4% (IC 95% 58,5–86,7%; 68 s, 25 n, 7 ?); acerto de classe 64,2% (48,9–78,4%); copas fundidas marcadas em 22 (14 s). Bootstrap estratificado B = 2000, semente 42; Wilson por estrato.
- Monografia: `resultados.tex` ganhou `\subsection{Conferência visual de uma amostra da área urbana}` (`sec:conferencia_cidade`, tabela `tab:conferencia_cidade`) antes da cobertura de dossel; o PENDENTE de `sec:georref` virou remissão; o parágrafo após `tab:cidade` cita a conferência e relativiza a confiança na palmeira. `main.tex`: frase da metodologia aponta para a nova seção; Conclusão ganhou uma frase com o resultado; nos trabalhos futuros, "conclusão da conferência das 100 árvores" virou "conferência de uma amostra maior por mais de um avaliador". Deslocamento não foi medido (dito no texto). Backup: `audit_backups_20261102/latex_antes_conferencia_100_20261121/`.

## 2026-11-21 — Pacote para a conferência das 100 árvores no QGIS

- `runs/conferencia_100_arvores_20261121/`: cópia dos 983 tiles da amostra + vizinhos (de tiles512), `mosaico_amostra.vrt` (caminhos relativos; GDAL do env `geo`, o do `siace` falha por libnetcdf), as camadas da amostra e `amostra_100_preencher.csv`, com `LEIAME.md`. A amostra original (`runs/amostra_verificacao_cidade_segformer_20261111/`) não foi alterada.

## 2026-11-21 — Figura do mapa de árvores volta a ser a do centro

- `resultados.tex`: `fig:mapa_resultado` passa a usar `mapa_centro.png` (`runs/mapa_centro_20261117/`, 531 árvores, visão + detalhe), como o autor usava no Overleaf; a versão local ainda apontava para `mapa_resultado_especies.png` (cidade inteira, pontos ilegíveis) e sobrescreveu a do Overleaf no último envio. Legenda e duas frases ajustadas (a seção de cobertura não cita mais as faixas brancas da figura). Backup: `audit_backups_20261102/resultados_antes_fig_centro_20261121.tex`.

## 2026-11-21 — Fontes de figuras e tabelas padronizadas

- Todas as 28 fontes (`\fonte{}`) em `main.tex`, `resultados.tex`, `trabalhos_relacionados.tex` e `tabela_tecnologias.tex` passaram a "Elaborado pelo autor (2026)." (tabela de trabalhos relacionados: "…, a partir dos trabalhos citados."). Setorização: "Elaborado pelo orientador, Prof. Rodrigo Smarzaro." (informado pelo autor; resolve o PENDENTE).
- Os caminhos de runs/scripts que estavam nas fontes foram para `SIACE_github/FONTES_FIGURAS_TABELAS.md` (rótulo → dados/script).
- Backup: `audit_backups_20261102/latex_antes_fontes_20261121/`.

## 2026-11-21 — Cobertura de dossel na monografia

- `referencias.bib`: Nucci e Cavalheiro (1999), GEOUSP n. 6 (30% recomendado e < 5% "deserto", Oke apud Lombardo; conferido no PDF); Konijnendijk (2023), regra 3-30-300 (dados do registro da Springer); Guo, Liu e Zhu (2024), cobertura de 5% a 34% em cidades brasileiras.
- `main.tex`: parágrafo "Cobertura de dossel" em Avaliação dos Resultados; uma frase no Resumo e no Abstract; frase na Conclusão.
- `resultados.tex`: subseção nova `sec:cobertura_dossel` (tabelas `tab:cobertura_validacao` e `tab:cobertura_classes`, figuras `fig:cobertura_dossel` e `fig:cobertura_ortofoto`), entre Aplicação à área urbana e Envio ao OSM.
- Mapas: legenda das células ocultas corrigida de "Sem dado" para "Borda (não avaliada)" — as 47 células ocultas são todas fragmentos com < 2.500 m² no perímetro; nenhuma por falta de imagem. Slide 16 atualizado com o mapa novo.
- Backup: `audit_backups_20261102/latex_antes_cobertura_dossel_20261121/`.

## 2026-11-21 — Resultados: cobertura de dossel da cidade e sementes da EfficientNet

- Cobertura (`runs/cobertura_dossel_20261120/resumo.json`): 13.360 tiles no perímetro, 18 min na GPU. Cobertura de copa 5,95% (20,5 ha em 344,3 ha); 374 de 421 células exibidas; mediana 3,2% por célula; 228 células < 5%, 91 de 5–10%, 34 de 10–20%, 12 de 20–30%, 9 ≥ 30%. Conferência independente: correlação 0,94 com a área das copas do `pipeline_cidade_segformer_20261111` por célula. Mapas `mapa_cobertura_dossel.png` e `mapa_cobertura_ortofoto.png`.
- Apresentação: slide 16 novo "Cobertura de dossel da área urbana" (20 slides; backup em `audit_backups_20261102/`).
- Sementes da EfficientNet (`runs/classif_seeds_efficientnet_20261120/resumo.json`): F1-macro de validação anotados 0,590 ± 0,016 × aug_treino 0,606 ± 0,020 (p = 0,26); teste F1-macro 0,632 ± 0,022 × 0,580 ± 0,026 (p = 0,057); acurácia balanceada 0,756 ± 0,042 × 0,634 ± 0,053 (p = 0,057). Pela regra 3 do critério, a média de validação de aug_treino é maior: registrar como limitação, sem trocar o modelo.

## 2026-11-20 — Cobertura de dossel em grade de 100 m (validação feita; cidade na fila)

- Critério fixado antes: `runs/cobertura_dossel_20261120/criterio.md`. Script novo `cobertura_dossel.py` (etapas `validar`, `cidade`, `mapa`), que usa as funções de segmentação do `pipeline_cidade.py`, o checkpoint e o limiar congelado da SegFormer MiT-B1. Cobertura = máscara binária, sem individualização.
- Os 5.079 tiles ausentes de `tiles512` estão em `~/proj/novas_imagens` (separados da pasta original; nenhum anotado; 25/25 idênticos ao zip). Entram só neste cálculo; os números da cidade e o envio ao OSM não mudam. No perímetro urbano (`Placemark.kml`): 13.360 tiles, 391 deles vindos de `novas_imagens`.
- Validação no teste (204 tiles, uma vez, CPU): IoU agregado de conferência 0,773 (igual à monografia); cobertura predita 19,95% × anotada 18,82%, viés relativo +6,0% (IC 95% +1,7% a +10,9%); por tile, erro médio +1,13 pp (IC 0,32 a 2,02), erro absoluto médio 3,29 pp, r = 0,934; por grupo de tiles vizinhos (78), erro absoluto médio 3,31 pp, r = 0,941.
- Etapa `cidade` + `mapa` na fila (`runs/cobertura_dossel_20261120.sh`, nohup), esperando os treinos da EfficientNet liberarem a GPU.
- Script novo `gerar_mapa_cobertura_ortofoto.py` (para a apresentação): grade semitransparente sobre o mosaico do perímetro (cache do mapa urbano + 391 tiles de `novas_imagens` que faltavam). Testado com células fictícias; a versão real é gerada depois da etapa `cidade`.

## 2026-11-20 — Repetição com sementes da EfficientNet-B0 (em execução)

- Critério fixado antes dos treinos: `runs/classif_seeds_efficientnet_20261120/criterio.md`. Os treinos novos só medem a variação entre treinos; o modelo do sistema continua `classif_20261102_163825_efficientnet_anotados` (semente 42) e nada no pipeline, na cidade ou no OSM muda.
- 6 treinos (variantes `anotados` e `aug_treino`, sementes 1, 2 e 3), com os mesmos argumentos dos runs de 02/11; `aug_treino` reutiliza o manifesto de cópias aumentadas do run original. Runs `runs/classif_seeds_20261120_efficientnet_<variante>_s<semente>`.
- O script mudou depois dos runs originais só para acrescentar `--binary-target`; o caminho multiclasse não mudou. Hashes dos scripts em `scripts_sha256.txt`.
- Cadeia: `runs/classif_seeds_efficientnet_20261120.sh` (nohup, PID em `.pid`, saída em `.out`). Estimativa: ~6 h.

## 2026-11-20 — Fórmula do IoU movida para os Métodos

- A equação do IoU estava só em `resultados.tex` (Configuração Experimental), embora a seção de métricas dos Métodos já a citasse. Agora fica em `main.tex` (Métricas de avaliação, segmentação binária), com a definição de A e B e do intervalo de 0 a 1; em `resultados.tex` ficou só a referência à Equação. Backup: `audit_backups_20261102/latex_antes_iou_metodos_20261120/`.

## 2026-11-20 — Recorte dos tiles com GDAL

- `main.tex` (arquivo atual da monografia; também em `main_3.tex`) (Software de Processamento de Imagens): o PENDENTE do orientador foi substituído por "recortado em *tiles* com a biblioteca GDAL", por decisão do autor, sem detalhar o comando. O software do ortomosaico não é nomeado (o GDAL não gera ortomosaico a partir das fotos). Backup: `audit_backups_20261102/latex_antes_gdal_20261120/`.

## 2026-11-20 — Revisão do texto (introdução e referencial teórico)

Backup: `audit_backups_20261102/latex_antes_revisao_texto_20261120/`. Sem mudança em métodos, números ou resultados.

- `main_3.tex`: introdução condensada (saem os 170 milhões de dólares da Comissão Europeia e o parágrafo longo sobre redes elétricas, mantido em uma frase); objetivo específico "CNNs" trocado por "aprendizado profundo" (a SegFormer é *transformer*); saem câmeras térmicas/COVID-19, o estudo de turbinas eólicas, a filtragem espacial/Fourier (não usadas), a variação de pinheiros; correção radiométrica encurtada e ligada à Seção de Preparação dos Dados; Andrade (2020) e Flavia reduzidos a um parágrafo; Aplicações Práticas e Discussão condensadas, com referência à Tabela de tecnologias; OSM da Seção de Métodos reescrito com o uso real (API 0.6, OAuth 2.0, `natural=tree`), sem Overpass; palavras-chave corrigidas; "Um drone também chamados" corrigido.
- `tabela_tecnologias.tex`: chaves BibTeX trocadas por `\textcite`; Pádua et al. (2020) passa de "Aérea (satélite)" para "Aérea (VANT)" (o título do artigo diz *UAV-based*); acentos.
- `siglas.tex`: removidas IANS, DEM, EO e 6S (sem uso).
- `referencias.bib`: entrada duplicada `erigo` fundida em `da2019metodologias` (url e editora); "visãoo", "Detecçãoo", "Iniciaçãoo"; Barbosa (2018) e Sá et al. (2021) com periódico/volume/páginas em campos próprios; "Accessed" → `urldate`; `howpublished` duplicado removido.

## 2026-11-18: filtro de altura (MDS − MDT no ponto) para falsas árvores — não adotado

- `filtro_altura.py` + `runs/filtro_altura_20261118/` (critério fixado antes em `criterio.md`). Alturas de `arvores com altura.geojson` (autor) ligadas às instâncias de validação/teste (idênticas às da cidade: 390/390 e 355/355).
- Validação: sem filtro F1 0,679 (390 inst., 268 TP); melhor limiar 1,50 m → 0,676; nenhum h de 0,25 a 3,00 m superou o sem filtro. Pelo critério, o filtro não foi adotado e o teste NÃO foi avaliado. Mesmo h = 0,25 m remove 5 acertos (altura do ponto central < 0,25 m em árvores reais).
- Adendo OSM (`osm_filtro_altura.py`, critério no mesmo `criterio.md`): grade s × h com meta de precisão ≥ 0,95 na validação e máximo de copas aprovadas. Escolhido h = 2,50 m, s = 0,87 (val: 264 copas, precisão 0,955, contra 245 e 0,951 do filtro atual). Teste (uma vez): 276 copas, precisão 0,931 (0,938 sem bananeira), 82% das árvores reais mantidas; filtro atual no teste: 233, 0,970, 72%. A precisão caiu abaixo da meta no teste; o filtro do OSM atual (sem altura) é mantido. Nada enviado.
- Pasta `treinos_altura_20261118/` criada para a ideia de altura como banda de entrada (aguarda raster MDS − MDT).

## 2026-11-18: ortomosaico e correções geométrica/radiométrica (banca da área de Geo)

- `main_3.tex`: o ortomosaico deixa de ser descrito como montado no QGIS; software da ortorretificação e do recorte dos tiles fica como PENDENTE visível para o orientador (seção `sec:sw_processamento`, nova label). A frase "nenhuma correção geométrica ou radiométrica" (seção OpenCV) foi substituída, na Preparação dos Dados, por: ortorretificação na geração do ortomosaico, sem ajuste por pontos de controle nem correção radiométrica, com justificativa e limitação de generalização. Seção QGIS reduzida à verificação dos resultados (o mapa da monografia é gerado por script).
- `gerar_fluxograma.py`: caixas sem "QGIS" ("Ortomosaico RGB (ortorretificado)"; "Mapa ... (GeoJSON / CSV)"); figura regenerada. Backup: `audit_backups_20261102/latex_antes_ortomosaico_20261118/`.

## 2026-11-17: imagens na pasta figuras/ do Overleaf

- `main_3.tex`: `\graphicspath{{figuras/}{./}}`. `resultados.tex`: removidos os `\IfFileExists` das 5 figuras (todas existem; o teste não usa o graphicspath). Backup: `audit_backups_20261102/latex_antes_pasta_figuras_20261117/`.

## 2026-11-17: monografia sem a seção de parâmetros e sem a limitação de GPU

- `main_3.tex`: removida a seção `sec:metodo_parametros` (texto, `tab:parametros` e `fig:parametros`); na justificativa dos modelos, removidas as menções à memória da GPU (picos de MiB, teste do MiT-B2, teste de variantes da EfficientNet) e às "restrições de hardware". Pedido do autor. Backup: `audit_backups_20261102/latex_antes_remover_parametros_gpu_20261117/`. Os runs e scripts de medição continuam no repositório.

## 2026-11-11: SegFormer MiT-B1 substitui a U-Net ResNet-34 no sistema

- Critério atendido (`runs/seeds_segformer_comparacao_20261110/decisao.txt`): média de 4 sementes, IoU val 0,765 vs 0,742 (+0,023, 4/4 sementes, p = 0,029) e IoU teste 0,761 vs 0,744 (+0,017, 3/4, p = 0,114). Modelo do sistema: `runs/arq_segformer_mit_b1_20261110` (semente 42, limiar 0,40). A U-Net vira modelo experimentado.
- `pipeline_cidade.py`: novos `--seg-arch`/`--seg-encoder` (padrão Unet/resnet34, comportamento original); arquitetura registrada no `provenance.json`. Teste com 8 tiles: máscaras idênticas às da avaliação de teste da SegFormer.
- Cadeia `runs/refazer_segformer_20261111.sh` (nohup): etapas 1–5 de `refazer_corrigido_20261106.sh` com a SegFormer; saídas `runs/*_segformer_20261111/`. U-Nets multiclasse não refeitas (independem do modelo do sistema).
- `bootstrap_ic_segformer.py`: IC 95% (validação SegFormer x U-Net, detecção e classificação de ponta a ponta).
- Geradores apontados para a SegFormer: `gerar_curvas_p1.py`, `gerar_figuras_resultados.py`, `gerar_fluxograma.py`; `gerar_figura_parametros.py` inclui as 4 arquiteturas (`runs/parametros_modelos_20261111/`). Backups em `audit_backups_20261102/scripts_antes_segformer_20261111/` e `latex_antes_segformer_20261111/`.
- `referencias.bib`: `xie2021segformer` (NeurIPS 34, pp. 12077–12090, página oficial), `chen2018encoder` e `zhou2018unetpp` (Crossref).
- Individualização: a cadeia parou de propósito (rc = 3) porque, com as máscaras da SegFormer, uma variante do Watershed superou as componentes conexas na validação (`runs/individualizacao_val_segformer_20261111/resultado.csv`): h-maxima h = 0,75 m, a_min = 3000 → F1 0,679; CC 0,674; janela f = 251 0,662. O autor escolheu o h-maxima (`frozen_instance_params_hmax.json`).
- `separate_tree_crowns_whatershed.py`: `segment_tree_crowns_hmax` (idêntica a `calibrar_individualizacao_val.hmax_watershed`: 0 pixels de diferença, 390 instâncias nos 203 tiles de validação); `pipeline_cidade.py`: método `watershed_hmax`.
- Cadeia `runs/refazer_segformer_hmax_20261111.sh` (nohup, concluída 10:43, todas rc = 0): validação, teste (h-maxima, CC e janela), limiares OSM e cidade.
  - Teste de ponta a ponta: h-maxima F1 0,671 [0,623; 0,719], 355 instâncias, 0,85 inst./árvore; CC 0,662; janela 0,659; U-Net + CC 0,666 (dif. +0,005 [−0,039; +0,052]). Classificação e2e: acurácia 0,823, F1-macro 0,576.
  - OSM (`runs/osm_limiares_segformer_20261111`): seg_prob ≥ 0,91 (precisão val 0,951, 245 copas), área ≤ 217,4 m², nenhuma espécie liberada; teste 97,0% de 233 copas.
  - Cidade (`runs/pipeline_cidade_segformer_20261111`, 106 min): 22.879 copas → 13.491 árvores (4.284 uniões); 197 maciços > 500 m² (36% da área), maior 1,7 ha em 107 tiles.
- `osm_envio_dev.py`: etiqueta `source` do changeset passa a citar SegFormer MiT-B1. Simulação (`runs/osm_envio_simulacao_segformer_20261111`): 4.461 passam nos filtros, 99 bananeiras excluídas, 4.362 selecionadas (32,3%) em 9 lotes. Envio feito em 2026-11-11 a pedido do autor: 2.289 nós antigos apagados (changeset 723832); 4.362 nós criados (723833 com 5 de teste; 723836–723844); conferência pela API: 4.362 visíveis só com natural=tree, 2.289 antigos com visible=false.
- Nova amostra de 100 árvores para conferência no QGIS: `runs/amostra_verificacao_cidade_segformer_20261111/` (mesmo `sortear.py`, só a fonte mudou).
- Figuras regeneradas: `mapa_resultado_especies.png`, `watershed_exemplo.png`, `fluxograma_projeto.png` (etapa 2 = Watershed h-maxima).
- Monografia: resumo, abstract, metodologia, resultados (arquiteturas, individualização, ponta a ponta, cidade, OSM) e conclusão reescritos para a SegFormer + h-maxima; a U-Net ResNet-34 fica como modelo experimentado.

## 2026-11-10: repetição com sementes da SegFormer MiT-B1

- Decisão do autor: se a SegFormer se sobressair, ela substitui a U-Net ResNet-34 no sistema e na monografia; a U-Net passa a ser modelo experimentado.
- Critério definido antes dos resultados: a SegFormer substitui a U-Net se, nas sementes 42, 1, 2 e 3 (as mesmas da repetição da U-Net), a média do IoU de validação e a média do IoU de teste forem maiores que as da U-Net.
- `runs/seeds_segformer_20261110.sh` (nohup, log em `runs/seeds_segformer_20261110.out`): runs `runs/seeds_segformer_s{1,2,3}_20261110/`; depois `avaliar_seeds_segformer.py` → `runs/seeds_segformer_comparacao_20261110/` (`resultados_por_run.csv`, `resumo.csv`, `decisao.txt`), com teste de permutação exato como em `avaliar_seeds.py`.

## 2026-11-10: benchmark de arquiteturas de segmentação

- `teste_memoria_segmentacao.py` → `runs/teste_memoria_segmentacao_20261110/memoria.csv`: com 512 px, lote 6 e precisão mista, couberam U-Net++ ResNet-34 (3.890 MiB reservados), DeepLabV3+ ResNet-34 (1.836) e SegFormer MiT-B1 (3.656); SegFormer MiT-B2 estourou a memória.
- `tree_segmentation_arquiteturas.py`: cópia de `tree_segmentation_resnet_26_512softAug4.py`; muda só o modelo (`--arch`, `--encoder`) e define `CUBLAS_WORKSPACE_CONFIG=:4096:8`, exigido pelo modo determinístico na DeepLabV3+ e na SegFormer. O script original não foi alterado.
- `evaluate_seg_arquiteturas_test.py`: cópia de `evaluate_unet_binary_test.py` com `--arch`/`--encoder`. Conferido: reproduz pixel a pixel as predições de teste da U-Net principal.
- `avaliar_arquiteturas.py`: avalia o teste uma vez por modelo e resume métricas, IC 95% por bootstrap por tile e diferença pareada de IoU contra a U-Net (mesmas funções, B = 2000 e semente 42 de `bootstrap_ic_corrigido.py`). Reproduz o IoU da U-Net principal (0,751 [0,717; 0,781]).
- Testes de uma época em pasta temporária (fora de `runs/`): os três treinam, salvam e carregam o checkpoint.
- Treinos: `runs/arquiteturas_20261110.sh`, executado com `nohup` (log em `runs/arquiteturas_20261110.out`); runs `runs/arq_*_20261110/`; resumo em `runs/arquiteturas_comparacao_20261110/`.
- Resultado (os três treinos com rc=0, parada antecipada em todos; resumo em `resultados.csv` e `ic_95.csv`). IoU de validação / IoU de teste [IC 95%] / diferença pareada de IoU contra a U-Net [IC 95%]:
  - U-Net ResNet-34: 0,748 / 0,751 [0,717; 0,781].
  - SegFormer MiT-B1: 0,765 / 0,773 [0,739; 0,805] / +0,022 [+0,004; +0,040]; 21 épocas, 136 min.
  - DeepLabV3+ ResNet-34: 0,740 / 0,744 [0,707; 0,777] / −0,007 [−0,025; +0,009]; 16 épocas, 47 min.
  - U-Net++ ResNet-34: 0,726 / 0,723 [0,689; 0,756] / −0,028 [−0,047; −0,008]; 10 épocas, 38 min.
- Uma semente por arquitetura; a U-Net variou de 0,732 a 0,755 no teste entre sementes. Os modelos a jusante (detecção, classificação, OSM) continuam usando a U-Net ResNet-34.

## 2026-11-10: trabalhos relacionados alinhados ao sistema final

- `trabalhos_relacionados.tex`: o texto sobre individualização e a linha "Este trabalho" da tabela-síntese agora indicam componentes conexas, método adotado no lugar do Watershed (Seção `sec:ws_vs_cc`).
- `main_3.tex`: incluído `\input{trabalhos_relacionados}` logo após "Estudos Relacionados"; na conclusão, F1 do Watershed com máscara de referência corrigido de 0,607/0,859 para 0,608/0,862, como em `runs/watershed_ref_teste_420_20261106/resultado.json`.
- `main_3.tex`: nova seção `sec:metodo_parametros`, com texto, tabela e figura (`parametros_modelos.png`) dos parâmetros treináveis, gerados por `gerar_figura_parametros.py` a partir dos checkpoints e conferidos com as arquiteturas instanciadas (`runs/parametros_modelos_20261110/parametros.csv`).
- `main_3.tex`: nova seção `sec:metodo_escolha`, justificando U-Net, ResNet-34 e EfficientNet-B0 pela literatura e pelas restrições de dados e de GPU, e citação da ResNet-34. `referencias.bib`: entradas `he2016deep` e `huang2024tree` (Crossref) e páginas de `tan2019efficientnet` (PMLR). Memória medida por `teste_memoria_efficientnet.py` (`runs/teste_memoria_efficientnet_20261110/memoria.csv`).
- Backups: `audit_backups_20261102/latex_antes_trabalhos_relacionados_20261110/`, `latex_antes_input_trabalhos_relacionados_20261110/`, `latex_antes_tabela_parametros_20261110/` e `latex_antes_justificativa_modelos_20261110/`.

## 2026-11-09: monografia migrada para a classe abntex2-ufv v1.8

- `abntex2-ufv.cls` e `abntex2-ufv.sty` copiados do release 1.8 de https://github.com/Smarzaro/abntex2-ufv.
- `main_3.tex`: novo preâmbulo, palavras-chave com `\palavraschave` e `\keywords`, nota da folha de rosto com `\preambuloUFV{tcc}`, data da defesa, segundo `\addbibresource` (3 citações estavam sem referência).
- Títulos acima das 7 figuras, com o ambiente `ilustracao`; `tabela_tecnologias.tex` com título acima e fonte; `siglas.tex` com o ambiente `siglas` e VANT em português.
- Não compilado localmente (não há LaTeX na máquina). Imagens mantidas na raiz porque `\IfFileExists` não usa o `\graphicspath`.
- Backup e arquivos antigos (`abntex2-UFV.sty`, `brazil-abnt.lbx`): `audit_backups_20261102/latex_antes_abntex2ufv_18_20261109/`.

## 2026-11-08: envio ao servidor de testes do OSM e ajustes da monografia

- Removidos os 751 nós do modelo substituído (changeset 722097; registro em `runs/osm_remocoes_dev_registro.csv`).
- Enviadas 2.289 árvores de `runs/pipeline_cidade_corrigido_20261106`: changeset 722098 (5, teste) e 722099 a 722103. Conferido na API: 2.289 visíveis, só `natural=tree`.
- `main_3.tex`: retirada a menção ao critério de escolha dos 1.018 tiles (sem referência; 159 deles têm menos de 5% de pixels de árvore), citados `make_mask.py` e `make_mask_distin.py`, ajustados o envio ao OSM e a descrição do uso do OSM.
- `resultados.tex`: changesets e nós do envio; verificação das máscaras (1.018 reproduzidas exatamente: 707 de `labels/`, 311 de `labels2/`).
- Backups: `audit_backups_20261102/latex_antes_scripts_mascaras_20261108/` e `latex_antes_envio_osm_20261108/`.

## 2026-11-06: correção do alinhamento imagem-máscara na augmentation (branch `fix/aligned-augmentation`)

Baseline preservado: commit `e1f8ab5`, tag `baseline_before_alignment_fix`, arquivo `artifacts/legacy_run_20261106/` (somente leitura, com SHA256SUMS e MANIFEST.md).

### Corrigido
- **Crop aleatório desalinhado entre imagem e máscara.**
  - Problema: `random.seed()` não controla o sorteio do `torchvision.transforms.RandomResizedCrop`, que usa o gerador do PyTorch. Por isso, imagem e máscara recebiam recortes diferentes: 18 px de deslocamento mediano na escala 0,85–1,0 e 53 px na 0,4–1,0.
  - Correção: os parâmetros agora são sorteados uma única vez e aplicados aos dois (`aligned_augmentation.paired_random_resized_crop`).
- **Máscara reamostrada com interpolação bilinear.** Agora usa NEAREST, sem valores intermediários que virariam árvore na binarização.
- **Máscara de validação em `banchmark_Unet.py` e `Unet_plus_banchmark.py`.** O redimensionamento agora usa NEAREST.

### Arquivos modificados
- Novo módulo `aligned_augmentation.py`.
- Scripts de treino que passam a usar o crop pareado:
  - `tree_segmentation_resnet.py`
  - `tree_segmentation_resnet_26.py`
  - `tree_segmentation_resnet_26_512semAug.py` (código morto nesse script, sem efeito)
  - `tree_segmentation_resnet_26_512semAug2.py`
  - `tree_segmentation_resnet_26_512softAug2.py`
  - `tree_segmentation_resnet_26_512softAug4.py`
  - `tree_segmentation_resnet_26_512Aug5.py`
  - `tree_segmentation_resnet_26_256softAug2.py`
  - `banchmark_Unet.py`
  - `Unet_plus_banchmark.py`
  - `resnet3.py`
- Novos testes `tests/test_aligned_augmentation.py` (6 testes; limite configurável `ALIGN_MIN_IOU`).
- Novos scripts de auditoria e comparação: `verificar_alinhamento_visual.py`, `avaliar_alinhamento.py`, `comparar_e2e_alinhamento.py`, `runs/alinhamento_retreino_20261105_2215.sh` e `runs/alinhamento_avaliacao_20261105.sh`.
- `.gitignore`: passa a ignorar `artifacts/`.

### Não alterado (de propósito)
- Escalas, proporções, hiperparâmetros, split, protocolo de limiar e runs antigos.
- O modelo usado no pipeline, no mapa da cidade, no envio ao OSM e na monografia continua sendo `runs/p1_20261101_140519`.
- Achado documentado e ainda não corrigido: no softAug4, `valid_pixel_mask` é calculado depois do color jitter.

### Impacto medido
Detalhes em `runs/alinhamento_comparacao_20261105/RELATORIO_ALINHAMENTO.md`.
- Retreino das 6 configurações em `runs/alinhado_20261105_2215_*`. No modelo principal, o IoU de teste foi de 0,723 para 0,751, e o F1 de detecção de ponta a ponta de 0,631 para 0,666.
- Um retreino de controle sem mudança de código (512semAug) variou +0,020 de IoU. Com uma execução por configuração, o ganho não se distingue da variação entre treinos.
- Uma configuração (512softAug2) piorou −0,044.

### Repetição com sementes (2026-11-06)
- `--seed` (padrão 42, comportamento original) em `tree_segmentation_resnet_26_512softAug4.py`.
- Nova cópia do script do baseline `tree_segmentation_resnet_26_512softAug4_codigo_antigo.py`, com o crop desalinhado e só o `--seed` adicionado.
- `avaliar_seeds.py` (avaliação de teste e teste de permutação exato) e a cadeia `runs/seeds_alinhamento_20261106.sh`.
- Correção no `avaliar_seeds.py`: a pasta de saída agora é resolvida como caminho absoluto. Antes, um caminho relativo fazia o resumo falhar depois da avaliação.
- Resultado com 4 sementes de cada lado, no modelo principal:
  - IoU 0,725 → 0,744 (+0,020, maior nas 4 sementes, p = 0,11);
  - recall 0,879 → 0,905 (p = 0,029);
  - F1 de borda 2 px 0,184 → 0,247 (p = 0,029);
  - precisão sem mudança.
  - Detalhes no item 16 de `runs/alinhamento_comparacao_20261105/RELATORIO_ALINHAMENTO.md`.

## 2026-11-06: monografia refeita com a U-Net corrigida

Decisão do autor: a monografia passa a usar só os modelos treinados depois da correção do crop e não cita os anteriores. Os runs antigos ficam preservados, sem alteração.

### Modelo e protocolo
- Segmentação do sistema final: `runs/alinhado_20261105_2215_softAug4` (semente 42, limiar 0,40 escolhido na validação).
- Cadeia `runs/refazer_corrigido_20261106.sh`, com marcadores em `runs/refazer_20261106/`. Todas as etapas terminaram com rc=0:
  - máscaras de validação;
  - calibração da individualização: `calibrar_individualizacao_val.py`, com componentes conexas e min_area 3000 (F1 val 0,664). O h-maxima chegou a 0,662 e não superou;
  - ponta a ponta no teste, com componentes conexas (F1 0,666) e watershed (0,663);
  - limiares do OSM: seg_prob ≥ 0,91, nenhuma classe liberada para espécie;
  - cidade: 11.915 árvores, 89 min;
  - U-Nets multiclasse CE+Dice com `resnet3.py` corrigido (F1-macro 0,352; com inicialização binária, 0,280).

### Novos scripts e saídas
- `bootstrap_ic_corrigido.py`, com saída em `runs/bootstrap_ic_corrigido_20261106/`.
- `gerar_mapa_cidade.py`: mapa `mapa_resultado_especies.png`.
- `runs/watershed_ref_teste_420_20261106/reavaliar.py`. Reavalia o watershed nas máscaras de referência no teste com 420 árvores anotadas.
  - A diferença vem de `labels2/OrtoRP-3cm_082_013.json`, antes `... - Copia.json`, que agora é lido pela avaliação.
  - As instâncias são idênticas às do run de 2026-11-02 (804 e 360).
- `runs/amostra_verificacao_cidade_corrigido_20261106/`: nova amostra de 100 árvores sorteada da cidade nova. A amostra anterior não tinha sido preenchida.
- `runs/osm_envio_simulacao_corrigido_20261106/`: simulação com 2.289 árvores, todas apenas `natural=tree`.
- `osm_remover_envio_antigo_dev.py`: apaga do servidor de testes os 751 nós enviados a partir de `runs/pipeline_cidade_segprob_20261105`. O padrão é só simular; o envio é executado pelo autor.

### Arquivos de figuras e geradores
- `gerar_curvas_p1.py`, `gerar_figuras_resultados.py` e `gerar_fluxograma.py` passam a usar o modelo corrigido e o limiar 0,40.

### LaTeX
- `resultados.tex` e `main_3.tex` (resumo, abstract e conclusão) reescritos com os números novos. Backup do estado anterior em `audit_backups_20261102/latex_antes_modelo_corrigido/`.
