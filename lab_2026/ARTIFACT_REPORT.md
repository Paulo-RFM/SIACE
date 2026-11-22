# ARTIFACT_REPORT — Auditoria SIACE (2026-11-02)

## Resumo

1. **Segmentação.** O run P1 (`runs/p1_20261101_140519/`, não modificado) é consistente. Tem split espacial com hash `8adf3e72…b658`, limiar 0,65 escolhido na validação e congelado antes do teste, e teste avaliado uma única vez: IoU 0,723 e F1 0,839. Uma segunda execução reproduziu o resultado bit a bit. A monografia, porém, descrevia o protocolo antigo (70/30 aleatório, só validação), com números que **não batem** nem com o CSV antigo (`modelos70_30/benchmark_unet.csv`).
2. **Espécies.** O classificador histórico tinha vazamento grave e foi descartado. Os pseudorrótulos existentes são re-rótulos automáticos dos próprios recortes manuais. O pipeline foi refeito com split herdado do tile, validação e teste só com anotados, bloqueios explícitos e um run auditável. Os dados atendem ao critério mínimo (4 espécies com ≥ 10 amostras de teste). O smoke rodou. **O treino completo não foi executado** e aguarda as decisões D1–D3 do `EXECUTION_PLAN.md`.

## Inconsistências da monografia

As linhas indicadas são as do arquivo atual. Os originais estão em `audit_backups_20261102/`.

| # | Arquivo:linha | Inconsistência | Ação | Justificativa |
|---|---|---|---|---|
| 1 | `main_3.tex:180` (resumo), `:192` (abstract) | Descreviam Random Forest, SVM, câmera 360°, imagens de solo e "grande conjunto de dados" como trabalho realizado; nada disso existe no repositório | **Reescritos** com o trabalho real e números do P1; métricas de espécies declaradas pendentes | Divergência entre resumo, métodos e resultados |
| 2 | `main_3.tex:253` | A introdução prometia integrar imagens de rua, o que contradiz o objetivo geral (só aérea) | Reescrito; imagens de rua passam a ser trabalho futuro | Não há código nem dados de rua |
| 3 | `main_3.tex:288` | Objetivo citava AUC, precisão e recall em "teste previamente rotulado" | Alinhado às métricas efetivamente usadas e ao teste espacial | Coerência com os resultados |
| 4 | `main_3.tex:293` | "Validar o sistema em cenário real", sem evidência | Reformulado como aplicação do pipeline, com comentário `PENDENTE` | Não existe saída geoespacial nem validação de campo |
| 5 | `main_3.tex:483` | Placeholder "precisa de uma introdução para o capítulo???" | Introdução do capítulo de Métodos escrita | Placeholder |
| 6 | `main_3.tex:500-515` (antes ~497-510) | Métodos descreviam correção radiométrica, Fourier e Otsu no futuro, técnicas que não foram usadas | **Substituídos** por Preparação dos Dados, Divisão Espacial, Segmentação (benchmark, ResNet-34 como codificador comum, limiar na validação), Watershed e Classificação | Metodologia divergente do código |
| 7 | `main_3.tex:519` (antes "TensorFlow", "Keras") | Nenhum script usa TensorFlow ou Keras | Substituídos por uma subseção PyTorch; Keras removida | Verificado por grep em `lab_2026/*.py` |
| 8 | `main_3.tex:449` | "classificação de falhas" | "classificação de árvores" | Erro de conteúdo |
| 9 | `main_3.tex:458,465,539` | Referencial no futuro com TensorFlow; subseção OSM descrevia a U-Net | Corrigidos; OSM declarado como não utilizado | Nenhum código usa OSM/Overpass |
| 10 | `main_3.tex:551` | A avaliação falava em "acurácia de detecção" genérica | Métricas reais de segmentação e classificação, com teste espacial | Coerência |
| 11 | `main_3.tex:569` | **Não havia capítulo de Conclusão** | Conclusão adicionada, limitada a fatos rastreáveis, com `\pendente` para espécies | Estrutura exigida |
| 12 | `resultados.tex` (antigo l.15) | "70% treino / 30% validação, random_state 42" | Substituído pelo split espacial 60/20/20 (611/203/204) | O protocolo antigo não é o atual |
| 13 | `resultados.tex` (antigo l.43-64, Tabela de encoders) | Valores sem fonte e divergentes do CSV (ex.: ResNet-34 com lr 1e-4 = 0,706 no texto contra 0,742 no CSV); conclusão "EfficientNet melhor, ResNet-34 pior" invertida em relação ao CSV; protocolo 70/30 de validação | **Removida.** Substituída pela Tabela do P1 (teste) | Números não rastreáveis |
| 14 | `resultados.tex` (antigo l.64) | Justificava o uso da EfficientNet-B0 no classificador pelo benchmark de encoders da U-Net | Removido | Confundia codificador de segmentação com classificador |
| 15 | `resultados.tex` (antigo l.72-91, Tabela res/aug) | Linhas de 256 px (todos os tiles são 512), validação apresentada como desempenho, IoU 0,749 sem fonte nos runs atuais | Substituída pela tabela dos runs com limiar 0,5, rotulada como **não diretamente comparável** | Mistura de protocolos e limiares |
| 16 | `resultados.tex` (antigo l.97) | "pico na época 17, IoU 0,749" | P1: melhor época 12 (IoU val 0,722 @0,5), parada na 17 | `metrics.csv` do P1 |
| 17 | `resultados.tex` (antigo l.103) | Figura `curvas_treinamento_seg.png` inexistente | Gerada por `gerar_curvas_p1.py` a partir do P1 (PNG + CSV) | Dado real |
| 18 | `resultados.tex` (antigo l.123,165,183) | Figuras `watershed_exemplo.png`, `matriz_confusao_especies.png` e `mapa_resultado_especies.png` inexistentes, o que quebraria a compilação | Envolvidas em `\IfFileExists` com caixa `\pendente` | Compila sem inventar figura |
| 19 | `resultados.tex` (antigo l.22, l.28) | `\cite{jadon2020survey}`, `ronneberger2015u` e `tan2019efficientnet` ausentes do `.bib` | Frase do jadon removida; as duas entradas canônicas foram adicionadas ao `referencias.bib`, com comentário para o autor conferir a paginação | Citações quebradas |
| 20 | `resultados.tex` (antigo l.132-136) | Afirmava que os recortes vinham das instâncias de watershed (padding 16), mas vêm dos polígonos manuais (padding 20); dizia que pseudorrótulos (conf > 0,5) e o balanceamento até 1200/800 fizeram parte do treino; split 80/20 estratificado | Reescrito com o protocolo real; o modelo histórico é declarado descartado por vazamento | Pseudorrótulos e aumento pré-split invalidam a avaliação |
| 21 | `resultados.tex` (antigo l.140-158) | Tabela de métricas de espécies com `% TODO` | Removida; um `\pendente` explícito aponta o arquivo que deve fornecer os valores | Não existem métricas válidas |
| 22 | `resultados.tex` (antigo l.174-176) | Descrevia o produto GeoPackage como resultado obtido | Descrito como implementação; declarado que não há saída gerada e que o script usa checkpoints antigos | Nenhum `.gpkg`/`.geojson` localizado |
| 23 | `resultados.tex` (antigo l.13) | Configuração de hardware genérica ("GPU NVIDIA, PyTorch 2.x") | Valores de `config.json` do P1 | Rastreável |
| 24 | `tabela_tecnologias.tex:13,16` | Linha "Novo" (aérea/terrestre, 360º) e legenda placeholder "Descrição da tabela" | Linha "Este trabalho" corrigida; legenda real; rótulo `tab:tecnologias` | Divergente do trabalho real |

Referências cruzadas: após as edições, todos os `\ref` têm `\label` correspondente e todos os `\cite` existem no `.bib` (checagem estática). Não há `TODO`, `INSTRUÇÃO` ou `???` remanescentes. **A compilação não foi testada**: não há LaTeX instalado nesta máquina.

## Correções de código

Ver `SPECIES_AUDIT_REPORT.md` para arquivo e linha. Em resumo:

| Arquivo | Tipo | Mudança |
|---|---|---|
| `species_data.py` | **novo** | Resolve e valida o split; esquema canônico; bloqueios (split ausente ou incompleto, tile em várias partições, tile fora do split, colunas ausentes, origem ausente, aumentado fora do treino ou órfão, pseudorrótulo fora do treino, sem proveniência ou só auto-predito, val/test com origem não anotada); relatórios |
| `species_metrics.py` | **novo** | Métricas globais e por classe, matrizes (3 normalizações, PNG + CSV), classification_report, confiança, ECE, acurácia × limiar, curvas PNG + CSV |
| `build_species_manifest.py` | **novo** | Manifesto auditável a partir dos JSONs, reaproveitando recortes verificados e gerando os de `labels2` em `species_crops_v2/` |
| `train_species_efficientnet.py` | reescrito | Portão de auditoria, `--audit-only`, `--smoke`, variantes, run em `runs/<run_id>/`, avaliação de teste única, `config.json` completo |
| `augment_rare_species.py` | reescrito | Só treino, com `parent_instance_uid` e relatório antes/depois |
| `separate_tree_crowns_whatershed.py` | corrigido | `indices=True` removido; proveniência obrigatória; contagem por tile |
| `extract_zoomed_species_crops.py` | corrigido | `pause()` removido; não sobrescreve mais `species_crops/`; esquema alinhado |
| `pipeline_segment_classify_to_shapefile.py` | corrigido (l.196) | Apenas `indices=True`; a troca de checkpoints fica com o autor |
| `gerar_curvas_p1.py` | **novo** | Figura das curvas da monografia a partir do P1 |
| `tests/test_species_pipeline.py` | **novo** | 25 testes |

Nada foi apagado ou movido. `runs/p1_20261101_140519/` e os demais runs históricos não foram alterados. `species_crops/`, `species_crops_balanced/`, `dataset_auto/` e `species_model/` continuam intactos. Backups em `audit_backups_20261102/`.

## Pendências

Ver a tabela D1–D5 e os passos 4 a 10 do `EXECUTION_PLAN.md`. As que bloqueiam texto:
- Métricas de espécies: dependem do passo 4 e de D1/D2.
- Nomes das espécies: D2.
- Pipeline integrado e mapa: passo 9 e ortomosaico.
- Watershed sobre o P1 e sua avaliação: passo 8.
- Comparação justa entre configurações de segmentação: reavaliar as configurações legadas com limiar selecionado na validação.
- Compilação do PDF e conferência das entradas bibliográficas adicionadas.

## Segunda rodada (2026-11-02): nomes das classes e suíte de classificação

**Decisões do autor aplicadas.** `labels2/` usa a mesma tabela de `labels/`. Nomes em `species_names.json`: árvore sem identificação, palmeira, árvore seca, mangueira, bananeira e jabuticabeira.

**Achado crítico.** `resnet2.py` e `resnet3.py` usavam uma paleta que não corresponde às máscaras. A classe 1 virava fundo e as classes 5 e 6 estavam trocadas (detalhes em `SPECIES_AUDIT_REPORT.md` §7). Os modelos multiclasse históricos são inválidos.

| Arquivo | Tipo | Mudança |
|---|---|---|
| `species_names.json` | novo | Nomes das classes 1–6 |
| `run_classification_suite.py` | novo | Orquestrador análogo a `run_training_suite.py`: pré-voo, `--dry-run`, `--only`, `--smoke`, `--experiments`, `suite_log.txt`/`suite_command.json`/`failure.json` por run, resumo e índice |
| `run_legacy_multiclass_training.py` | novo | Adaptador dos scripts multiclasse ao split espacial (como `run_legacy_binary_training.py`), com inicialização opcional pelo P1 e `config.json` |
| `evaluate_multiclass_test.py` | novo | Teste único por pixel: CM global, métricas por classe, PNG + CSV |
| `build_classification_index.py` | novo | `index_classification.csv` |
| `species_data.py` | ampliado | `SPECIES_PALETTE`, `rgb_mask_to_class` (falha com cor desconhecida), `check_mask_palette` |
| `resnet2.py`, `resnet3.py` | corrigidos | Paleta real; métricas por confusão acumulada e por classe; LR e tempo por época; pesos de classe só do treino (resnet3) |
| `train_species_efficientnet.py` | ajuste | Aceita o diretório de run criado pela suíte, mas continua recusando se já houver `config.json` ou checkpoint |
| `build_run_index.py` | ajuste | Ignora runs de classificação no `index.csv` binário |
| `resultados.tex`, `main_3.tex` | texto | Nomes das classes; "seis classes de copa" (duas não são espécies); parágrafo sobre a U-Net multiclasse e o descarte dos modelos antigos; nova pendência de métricas |
| `tests/test_species_pipeline.py` | ampliado | 30 testes (paleta, scripts com a paleta real, seleção de experimentos, nomes) |

Validações: `py_compile` OK; 30/30 testes; `run_classification_suite.py --dry-run` com pré-voo completo OK (manifesto, nomes, 1018 máscaras multiclasse casando com as imagens, paleta, checkpoint P1, disco); avaliador multiclasse testado em 3 tiles com um checkpoint histórico, com saída no scratchpad e fora de `runs/`. **Nenhum treino executado.**

Observação fora do escopo: `run_training_suite.py:295` grava `summary.json` dentro do diretório do **último** run de segmentação (variável `run_dir` do laço), o que explica `summary.json` em `p1_*`, `smoke4_*` e `treino_*_512softAug4`. Não foi alterado.

## Terceira rodada (2026-11-02): polígonos sem `group_id`

- Decisão do autor: polígono de árvore sem `group_id` = classe 1 (árvore sem identificação). É a mesma regra já usada pelas máscaras (`make_mask_distin.py`).
- `build_species_manifest.py`: deixa de descartar esses polígonos e grava `species_rule` (`group_id`, `group_id_ausente->1`, `correcao_11->1`). `generate_crops_from_json.py` recebeu a mesma regra, além do 11→1.
- Manifesto reconstruído: 2095 → **2113** recortes (+18 na classe 1: treino +12, val +5, teste +1), 1015 tiles, sem vazamento. Manifesto anterior em `audit_backups_20261102/species_dataset_v2_antes_null/`.
- Atualizados: `resultados.tex` (texto e tabela), `main_3.tex` (resumo, abstract, conclusão), `tabela_tecnologias.tex`, `INVENTORY_REPORT.md`.
- O arquivo `labels2/OrtoRP-3cm_082_013 - Copia.json` ainda existe no disco e continua excluído pelo manifesto.
