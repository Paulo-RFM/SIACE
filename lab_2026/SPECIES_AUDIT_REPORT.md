# SPECIES_AUDIT_REPORT — Pipeline de classificação de espécies

Data: 2026-11-02. As linhas indicadas como "orig." referem-se às cópias em
`audit_backups_20261102/`; as demais, ao arquivo atual.
Status possíveis: **OK**, **PROBLEMA**, **CORRIGIDO** (era problema e foi corrigido) e **NÃO VERIFICÁVEL**.

## 1. Onde estão definidos os artefatos

| Artefato | Arquivo:linha | Status | Evidência | Impacto |
|---|---|---|---|---|
| CSV de recortes manuais | `generate_crops_from_json.py:14-16,101-110` | OK | O esquema `image,tree_id,crop,species` do `species_crops/species_labels.csv` coincide com o script, as 1530 linhas casam 1:1 com os polígonos de `labels/*.json` e 25/25 recortes regenerados saíram idênticos | Os rótulos são manuais (LabelMe `group_id`) |
| Extração por instância (watershed) | `extract_zoomed_species_crops.py` orig. :15,87,129-134 | CORRIGIDO | `pause()` inexistente (NameError na l.87 orig.); gravava em `species_crops/species_labels.csv`, o que **sobrescreveria** o CSV manual; colunas `crop_path/label/image` divergentes | O script nunca rodou com sucesso (`masks_instances/` está vazio). Destino trocado para `species_crops_watershed/` e esquema alinhado, sem coluna `origin` de propósito |
| Máscaras de instância | `separate_tree_crowns_whatershed.py` orig. :20-23,65-70 (`indices=True` na l.67) | CORRIGIDO | `peak_local_max(indices=True)` foi removido no scikit-image ≥0.20 (instalado: 0.25.2), o que causaria TypeError; não registrava a proveniência | Agora exige `--mask-provenance` e o checkpoint, e grava `watershed_provenance.json` (hash, limpeza, footprint, área mínima) e `instances_per_tile.csv` |
| Máscara binária predita | `generate_binary_masks.py:15-17` | PROBLEMA (documentado) | Usa o checkpoint `unet_binary_tree_detector.pth` (sha256 `6fe2cc47…7731`); `masks_bin_pred/` tem data de 2025-11-24, anterior ao P1 (2026-11-01) | As máscaras preditas existentes **não** são do P1; nenhum recorte de treino depende delas |
| Máscaras de espécie | `extract_zoomed_species_crops.py:13` (`masks_distin/`, 707 arquivos), `make_mask_distin.py` | NÃO VERIFICÁVEL | Não há registro de qual script e de qual versão gerou `masks_distin/` | Irrelevante para o manifesto novo (os rótulos vêm do JSON) |
| CSV balanceado | `augment_rare_species.py` orig. :11-24,66-110 | CORRIGIDO | O aumento era aplicado ao CSV inteiro, antes do split: 1317 de 3670 cópias vêm de tiles de val/test | Vazamento. A nova versão só aumenta o treino e liga cada cópia a `parent_instance_uid` |
| Modelo EfficientNet-B0 histórico | `species_model/efficientnet_b0_best.pth` (sha256 `48de1082…be3`); treino orig. em `modelos70_30/train_species_efficientnet.py.save:65` | PROBLEMA | `train_test_split(df_balanceado, 0.2, stratify=species)` aleatório sobre os dados aumentados: cópias da mesma árvore em treino e validação | Qualquer métrica desse modelo está inflada. Descartado |
| Pseudorrótulos | `create_pseudo_dataset_classes.py:16-25,81-152` | PROBLEMA | `input_dir = species_crops/`: o classificador histórico re-rotula os próprios recortes manuais (conf ≥ 0,92), espécie 100% auto-predita, 509 de 1233 em tiles de val/test, sem `origin`/hash | Bloqueados pelo novo treino |
| Pseudorrótulos (tiles) | `generate_balanced_pseudo_dataset.py:22,29-32,203` | PROBLEMA | Classificador histórico + `unet_binary_tree_detector.pth`; `CONF_THRESHOLD = 0.5`; 341 de 344 tiles fora do split | Bloqueados (sem proveniência, sem tile no split) |
| Pseudomáscaras de segmentação | `make_pseudo_label_512.py:17-28` | OK (fora do escopo) | Checkpoint `unet_binary_tree_detector512_semAug.pth`, limiar 0,8; aplica-se a `novas_imagens/` | Não alimenta a classificação |
| Checkpoint do pipeline de inferência | `pipeline_segment_classify_to_shapefile.py:44-45,63,196` | PROBLEMA + CORRIGIDO parcial | Segmentação com o checkpoint antigo e limiar 0,5 (não o P1/0,65), classificador histórico, `indices=True` (quebra, corrigido na l.196) e ortomosaico `/home/paulomaciel/proj/ortomosaico.tif` inexistente | O pipeline integrado não pode ser descrito como resultado. A troca de checkpoint fica para o autor (ver EXECUTION_PLAN) |
| Correção de rótulo | `fix_species.py:11` | OK | Mapeamento `11 → 1` (1 ocorrência) | Reaplicado e registrado em `species_raw` |

## 2. Contagens

Ver `INVENTORY_REPORT.md`. Resumo:
- Anotados: 2095, sendo 1530 de `labels/` e 565 de `labels2/`.
- Aumentados válidos: 0. Os 3670 históricos foram rejeitados.
- Pseudorrótulos válidos: 0. Os 1233 + 344 históricos foram rejeitados.

## 3. Vazamento

| Verificação | Histórico (`species_crops_balanced` + split aleatório) | Novo (`species_dataset_v2`) |
|---|---|---|
| Mesmo tile em treino e val/test | Sim: split aleatório por recorte | 0 |
| Cópia aumentada da mesma árvore em mais de uma partição | Sim | 0 (sem aumentados; regra testada) |
| Aumentados derivados de tiles de val/test | 1317 | 0 |
| Pseudorrótulos em val/test | 509 (`pseudo_full`) | 0 |
| Tiles só na diagonal entre partições | — | 53 (limitação do split da segmentação) |

## 4. Cadeia com a segmentação

| Pergunta | Resposta | Evidência |
|---|---|---|
| Checkpoint usado para gerar máscaras de espécie/treino | **Nenhum**: os recortes de treino/val/teste vêm de polígonos manuais | `build_species_manifest.py`, `mask_provenance = manual_polygon_labelme` |
| Watershed aplicado em máscara de referência ou predita? | Default `masks_bin_pred/` (predita, checkpoint antigo); nenhuma saída existente | `masks_instances/` vazio |
| Recortes de teste vêm da mesma distribuição do pipeline real? | **Não**: bbox do polígono manual, padding 20, contra instância de watershed predita, padding 16 | `generate_crops_from_json.py:21` vs `extract_zoomed_species_crops.py:19` |
| P1 localizado? | Sim: `runs/p1_20261101_140519/checkpoint_best.pth` (sha256 `a5f78fe4…b616`), limiar congelado 0,65 (`frozen_threshold.json`), teste avaliado uma vez (`log.txt`) | Run não modificado |

## 5. Consistência de nomes de colunas

| Script | recorte | tile | espécie | origem |
|---|---|---|---|---|
| `generate_crops_from_json.py` | `crop` | `image` | `species` | — |
| `extract_zoomed_species_crops.py` (orig.) | `crop_path` | `image` | `label` | — |
| `augment_rare_species.py` (orig.) | `crop` | `image` | `species` | `origin`/`source_origin` só nos aumentados |
| `create_pseudo_dataset_classes.py` | `image` (caminho!) | — | `species` | — |
| `train_species_efficientnet.py` (orig.) | `crop` | `tile/source_tile/image` | `species` | `origin` **obrigatório**, mas ausente em todos os CSVs existentes, então o script falhava em qualquer CSV |
| **Esquema canônico novo** (`species_data.py`) | `crop_path` | `tile` | `species` | `origin` ∈ {anotado, aumentado, pseudorrotulo} |

`species_data.normalize_columns` aceita os aliases históricos, mas **nunca** infere `origin`.

## 6. Problemas adicionais no treino original

| Arquivo:linha (orig.) | Status | Problema |
|---|---|---|
| `train_species_efficientnet.py:166-168` | CORRIGIDO | `inv_freq[x-1]` pressupunha classes 1..6 contíguas e todas presentes no treino |
| `train_species_efficientnet.py:46-49` | CORRIGIDO | `use_deterministic_algorithms(True)` sem `warn_only` com EfficientNet (o backward de `adaptive_avg_pool2d` em CUDA não é determinístico) e sem `CUBLAS_WORKSPACE_CONFIG`, o que causaria RuntimeError |
| `train_species_efficientnet.py:277-335` | CORRIGIDO | Avaliação final só na validação; não existia avaliação de teste |
| `train_species_efficientnet.py:28` | CORRIGIDO | Saída fixa em `lab_2026/species_model/`, sem run auditável |

## 7. Segunda rodada (2026-11-02): scripts de classificação multiclasse por pixel

As linhas "orig." referem-se a `audit_backups_20261102/resnet2.py` e `resnet3.py`.

| Arquivo:linha | Status | Evidência | Impacto |
|---|---|---|---|
| `resnet2.py` orig. :34-42, `resnet3.py` orig. :35-43 (`color_to_class`) | **CORRIGIDO (crítico)** | A contagem real de cores em `lab_2026/masks_distin2/` (1018) e em `masks_distin/` (707) mostra exatamente {preto, cinza 128, verde, azul, amarelo, ciano, magenta}; vermelho (255,0,0) **não ocorre**. Os scripts mapeavam vermelho→1 (então cinza, que é a classe 1 com 48 M px, virava fundo), ciano→6 e magenta→5 (trocados). A paleta correta é a de `make_mask_distin.py:17-25` | Os modelos multiclasse históricos (`unet_tree_segmentation_model*.pth`) foram treinados com rótulos errados. Num teste de código em 3 tiles, `unet_tree_segmentation_model3.pth` previu 100% fundo (F1 = 0 na classe 1) |
| `resnet2.py` orig. :152-193, 226-243; `resnet3.py` orig. :206-238, 279-287 | CORRIGIDO | Métricas por lote seguidas de média dos lotes; classe ausente no lote conta 0 | Métricas de validação enviesadas para baixo. Agora uma matriz de confusão é acumulada sobre toda a validação, com macro só nas classes presentes e IoU/F1 por classe em cada época |
| `resnet3.py` orig. :176 | CORRIGIDO | `compute_class_weights(mask_train + mask_val)` | Os pesos da loss usavam estatística da validação. Agora usam só o treino |
| `resnet2.py` orig. :113, `resnet3.py` orig. :161 (train_test_split 70/30) | CORRIGIDO via adaptador | Divisão aleatória por tile, sem teste | `run_legacy_multiclass_training.py` impõe train/val do split espacial; o teste é avaliado à parte |
| `resnet2.py:144-146` | CORRIGIDO via adaptador | Retomava silenciosamente de `unet_tree_segmentation_model_aug_bias.pth` se existisse | No adaptador, `checkpoint_path` aponta para o run novo, que ainda não existe |
| `resnet2.py:23,284`, `resnet3.py:26` | CORRIGIDO via adaptador | Sobrescreviam `.pth` históricos em caminhos fixos | `torch.save` redirecionado para `<run>/checkpoint_best.pth` |
| `resnet2.py`/`resnet3.py` (CSV de histórico) | CORRIGIDO | Gravavam `unet_multiclass*_history.csv` na raiz de `lab_2026/`, sem LR, sem tempo (resnet3) e sem métricas por classe | Agora: `<run>/metrics.csv` com `lr`, `epoch_time_s`, `iou_<classe>` e `f1_<classe>` |
| `resnet2.py:132` | OK (documentado) | `encoder_weights=None`: o encoder não é pré-treinado | Diferença de configuração entre `unet_mc_ce` e `unet_mc_ce_dice` (além da loss) |
| `make_mask_distin.py:10` | NÃO VERIFICÁVEL | Grava em `/home/paulomaciel/proj/masks_distin2/`, que não existe; o diretório usado é `lab_2026/masks_distin2/` (2026-11-01, inclui `labels2`) | Proveniência das máscaras multiclasse pendente (D4) |
| `catboost_final.py` | FORA DO ESCOPO | Lê `train.csv` com `f0..f523` e alvo "attack" | Não é classificador de espécies; excluído da suíte |
