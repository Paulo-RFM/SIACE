# RELATORIO_METRICAS — Rastreabilidade de métricas

`<run>` = `runs/species_<variante>_<AAAAMMDD_HHMMSS>/`, gerado por `train_species_efficientnet.py`.
`<s>` ∈ {`val`, `test`}. O teste é avaliado uma única vez, com `checkpoint_best.pth`, que é escolhido por `val_f1_macro`.
Funções em `species_metrics.py` (SM) e `species_data.py` (SD).

## Classificação de espécies

| Métrica | Arquivo | Função / script | Conjunto | Definição |
|---|---|---|---|---|
| accuracy | `<run>/<s>_metrics_global.{csv,json}` | `SM.global_metrics` | val, test | acertos / N |
| balanced_accuracy | idem | idem (`balanced_accuracy_score`) | val, test | média do recall das classes presentes em y_true; `null` se houver só 1 classe |
| f1_macro | idem | idem | val, test | média simples do F1 das K=6 classes (classe ausente conta F1 = 0) |
| f1_macro_present_classes | idem | idem | val, test | F1-macro só sobre as classes presentes no conjunto |
| f1_weighted | idem | idem | val, test | F1 por classe ponderado pelo suporte |
| precision_macro / recall_macro | idem | idem | val, test | média simples por classe, `zero_division=0` |
| cohen_kappa | idem | `cohen_kappa_score` | val, test | (p_o − p_e)/(1 − p_e) |
| number_of_samples / number_of_classes / ..._present | idem | idem | val, test | contagens |
| loss | idem | `run_epoch` | val, test | CrossEntropy média por amostra |
| ece | `<s>_metrics_global.json`, `<s>_reliability.csv` | `SM.calibration` | val, test | Σ_b (n_b/N)·\|acc_b − conf_b\| em 10 bins da confiança máxima |
| precision, recall, f1, support por espécie | `<run>/<s>_metrics_by_class.csv` | `SM.per_class_metrics` | val, test | `precision_recall_fscore_support(labels=0..5)` |
| support_normalized | idem | idem | val, test | support / N |
| specificity | idem | idem | val, test | TN/(TN+FP), um contra todos |
| auc_ovr | idem | idem (`roc_auc_score`) | val, test | AUC um-contra-todos sobre a probabilidade softmax da classe; vazio se a classe não tiver positivos ou negativos |
| classification_report | `<run>/<s>_classification_report.txt` | `SM.text_report` | val, test | `sklearn.metrics.classification_report(zero_division=0, digits=4)` com nomes |
| Matriz de confusão (contagens, linha, coluna) | `<run>/<s>_confusion_{counts,row_normalized,col_normalized}.{csv,png}` | `SM.save_confusion` | val, test | linhas = referência anotada; colunas = predito |
| Confiança por amostra/classe | `<run>/<s>_confidence_per_sample.csv`, `<s>_confidence_by_class.{csv,png}`, `<s>_confidence_histogram.png` | `SM.save_confidence_outputs` | val, test | max softmax |
| Reliability diagram | `<run>/<s>_reliability.{csv,png}` | `SM.calibration` | val, test | acurácia × confiança média por bin |
| Acurácia × limiar de confiança | `<run>/<s>_accuracy_vs_confidence_threshold.{csv,png}` | `SM.accuracy_vs_threshold` | val, test | acurácia e cobertura das amostras com conf ≥ t. **O corte 0,5 é uma regra operativa, não uma garantia** |
| Predições | `<run>/<s>_predictions.csv` | `main` | val, test | rótulo, predição e probabilidade de cada classe por recorte |
| train/val loss, F1-macro, accuracy por época | `<run>/learning_curve_species.csv`, `curve_{loss,f1_macro,accuracy}.{csv,png}` | `SM.save_training_curves` | treino (amostrado pelo sampler, com augment), val | médias por época |
| F1 por classe por época | `<run>/curve_val_f1_per_class.{csv,png}` | idem | val | — |
| learning rate por época | `<run>/curve_learning_rate.{csv,png}` | idem | — | LR no início da época |
| effective_epoch, duração, pico de memória, parâmetros, versões, hashes | `<run>/config.json` | `main`, `environment_info` | — | leitura real do ambiente (`torch.cuda.get_device_properties`, `resource.getrusage`) |
| Contagens por espécie/partição, razão de desbalanceamento, vazamento, elegibilidade | `<run>/config.json → audit`, `species_dataset_v2/inventory.json` | `SD.counts_by_split`, `SD.imbalance_ratio`, `SD.leakage_report`, `SD.test_eligibility` | todos | razão = max/min das classes com n > 0 |

## Segmentação (citadas na monografia; arquivos não alterados)

| Valor na monografia | Arquivo | Script | Conjunto |
|---|---|---|---|
| Limiar 0,65; IoU val 0,726 | `runs/p1_20261101_140519/frozen_threshold.json`, `threshold_iou_curve.csv` | `tree_segmentation_resnet_26_512softAug4.py` | val |
| IoU 0,723; F1 0,839; P 0,805; R 0,876; Acc 0,937; 0,108 s/tile | `runs/p1_20261101_140519/test_eval/test_metrics.csv` | `evaluate_unet_binary_test.py` | test (204 tiles) |
| Melhor época 12 (IoU 0,722 @0,5); parada na 17 | `runs/p1_20261101_140519/metrics.csv` | idem | val |
| Reprodução idêntica | `runs/treino_20261101_203446_512softAug4/test_eval/test_metrics.csv` | idem | test |
| Tabela de configurações com limiar 0,5 | `runs/{20261101_185936_base,20261101_185936_26,20261101_185936_512semAug,treino_20261101_203446_512semAug2,treino_20261101_203446_512softAug2}/{metrics.csv,test_eval/test_metrics.csv}` | `run_legacy_binary_training.py` + `evaluate_unet_binary_test.py` | val, test |
| 1018 tiles; 611/203/204; 246 componentes | `unet_spatial_split/tiles_split.csv`, `tiles_by_split.json` | `build_unet_spatial_split.py` | — |
| 53 pares diagonais entre partições | recalculado nesta auditoria a partir de `tiles_split.csv` (linha/coluna do nome do tile) | — | — |
| Hardware/versões | `runs/p1_20261101_140519/config.json` | — | — |
| Figura `curvas_treinamento_seg.png` (+ `.csv`) | `runs/p1_20261101_140519/metrics.csv` | `gerar_curvas_p1.py` | val |
| Tabela de dados de espécies (2095 recortes) | `species_dataset_v2/inventory.json` | `build_species_manifest.py` | — |

## U-Net multiclasse por pixel (suíte de classificação)

`<run>` = `runs/<suite_run_id>_unet_mc_*`. Classes: 0 fundo, 1–6 conforme `species_names.json`.

| Métrica | Arquivo | Script | Conjunto | Definição |
|---|---|---|---|---|
| train_loss, val_loss, lr, epoch_time_s | `<run>/metrics.csv` | `resnet2.py`/`resnet3.py` via `run_legacy_multiclass_training.py` | treino/val | média por lote; LR no início da época |
| accuracy, precision, recall, f1, iou (por época) | `<run>/metrics.csv` | `metrics_from_confusion` | val | matriz de confusão acumulada na validação inteira; macro sobre as classes de árvore **presentes** (sem fundo) |
| iou_<classe>, f1_<classe> (por época) | `<run>/metrics.csv` | idem | val | por classe a partir da mesma matriz |
| pixel_accuracy | `<run>/test_eval/test_metrics.{json,csv}` | `evaluate_multiclass_test.py` | test (204 tiles) | diag(CM)/ΣCM, incluindo o fundo |
| miou_foreground_present, f1/precision/recall_macro_foreground_present | idem | idem | test | média sobre as classes 1–6 com suporte > 0 no teste |
| miou_all_classes | idem | idem | test | média das 7 classes (as ausentes contam 0) |
| precision, recall, f1, iou, support_px por classe | `<run>/test_eval/test_metrics_by_class.csv`, `test_classification_report.txt` | idem | test | a partir da CM global de pixels |
| Matriz de confusão (contagens, linha, coluna) | `<run>/test_eval/test_confusion_*.{csv,png}` | idem | test | linhas = referência |
| Pixels por classe por tile | `<run>/test_eval/test_pixels_per_tile.csv` | idem | test | suporte da referência |
| Configuração, pesos de classe, inicialização (hash do P1), ambiente | `<run>/config.json` | adaptador | — | leitura real do ambiente |
| Índice consolidado | `lab_2026/index_classification.csv` | `build_classification_index.py` | — | uma linha por run |

## Números adicionados à monografia em 2026-11-02 (noite)

| Valor | Arquivo-fonte | Script |
|---|---|---|
| EfficientNet anotados: val F1-macro 0,598; teste acc 0,835, bal-acc 0,741, F1-macro 0,639, F1-pond 0,853, kappa 0,645, ECE 0,067; por classe | `runs/classif_20261102_163825_efficientnet_anotados/{config.json,test_metrics_global.json,test_metrics_by_class.csv}` | `train_species_efficientnet.py` |
| EfficientNet aug offline: val 0,593; teste acc 0,871, bal-acc 0,634, F1-macro 0,603, F1-pond 0,876, kappa 0,687, ECE 0,064 | `runs/classif_20261102_163825_efficientnet_aug_treino/...` | idem |
| Figura `matriz_confusao_especies.png` | `.../efficientnet_anotados/test_confusion_counts.csv` | `gerar_figuras_resultados.py` |
| Watershed: busca na validação (12 combinações), f=251, a_min=1000; teste com referência (F1 0,607 com os parâmetros originais e 0,859 com os calibrados) | `runs/watershed_ref_calib_20261102/{val_sweep.csv,frozen_watershed_params.json,README.md}` | `separate_tree_crowns_whatershed.py` + `eval_watershed_vs_polygons.py` |
| Detecção de ponta a ponta no teste (P1 0,65 + watershed calibrado): P 0,676, R 0,584, F1 0,626, IoU 0,787, 361/418, MAE 0,81, exata 48,5% | `runs/pipeline_e2e_test_p1_20261102/{detection_test.txt,per_tile.csv}` | idem, sobre `runs/p1_20261101_140519/test_eval/predictions_test` |
| Figura `watershed_exemplo.png` (tile OrtoRP-3cm_049_022) | `runs/pipeline_e2e_test_p1_20261102/{in,out}` | `gerar_figuras_resultados.py` |
| Georreferência: GSD ≈ 3,2 cm; continuidade de 512 px; 360 instâncias → 273 árvores (57 uniões) | tags GeoTIFF dos tiles; `runs/watershed_ref_calib_20261102/georef_test_referencia/georef_summary.json` | `tree_georef.py` |
| Máscara do pipeline idêntica à do P1 (12 tiles verificados) | verificação em scratchpad, 2026-11-02 19:5x | `pipeline_cidade.py` |

## Adicionado em 2026-11-02 (noite, 2ª parte)

U-Net multiclasse (teste, pixels de 204 tiles; mIoU/F1-macro sobre as 6 classes de copa):
| run | perda | init | seleção | época | acc pixel | mIoU | F1-macro |
|---|---|---|---|---|---|---|---|
| unet_mc_ce | CE sem pesos | ImageNet | val_loss | 17/22 | 0,899 | 0,172 | 0,227 |
| unet_mc_ce_dice | 0,3 CE pond. + 0,7 Dice | ImageNet | val IoU | 10/15 | 0,728 | 0,243 | 0,366 |
| unet_mc_ce_dice_initP1 | 0,3 CE pond. + 0,7 Dice | P1 (276/278) | val IoU | 18/23 | 0,852 | 0,282 | 0,419 |
Fonte: runs/classif_20261102_163825_unet_mc_*/test_eval/.

Ponta a ponta no teste (runs/pipeline_test_20261102_205814/avaliacao_teste/): detecção P 0,676, R 0,584, F1 0,626
(244 TP / 361 pred / 418 GT; IoU médio TP 0,787); classificação nas 244 casadas: acc 0,803, bal-acc 0,592,
F1-macro 0,544, kappa 0,528. Regressão de máscara vs P1: 18 pixels diferentes em 204 tiles.
runs/pipeline_test_20261102_195654/ é um run interrompido (OOM no classificador, lote 32), sem resultados.

Correções no pipeline_cidade.py: lote do classificador 32 -> 8 (--cls-batch-size); tiles de borda menores que
512 (última coluna 158 px, última linha 380 px) completados com zeros para a rede e predição recortada de volta.

## Cidade (2026-11-03) — runs/pipeline_cidade_20261103_alpha
61.426 tiles; 28.775 sem dado (alpha=0); 28.937 copas -> 14.440 árvores (4.472 por união na borda); 5.596 s (GTX 1050 Ti).
Por classe: sem identificação 11.524; palmeira 987; mangueira 876; árvore seca 458; bananeira 389; jabuticabeira 206.
O run anterior runs/pipeline_cidade_20261102_205814 tratava a área transparente (branca, alpha=0) como imagem
(n_tiles_nodata=0; 14.410 árvores) e foi substituído; ficou preservado. Correção: load_tile usa alpha>0 como válido.

## Cidade com componentes conexas (2026-11-03/04)

Run: `runs/pipeline_cidade_cc_20261103/` (params `runs/watershed_vs_cc_val_20261103/frozen_instance_params.json`, CC min_area 3000).
Tiles 61.426 (28.775 sem dado; 32.651 com imagem); 19.982 instâncias; 10.251 árvores (2.894 por união entre tiles); 4.866 s (~81 min).

| Classe | Árvores | % | Área média (m²) | Confiança média |
|---|---:|---:|---:|---:|
| arvore_sem_identificacao | 8158 | 79,6 | 86,5 | 0,871 |
| palmeira | 755 | 7,4 | 22,8 | 0,771 |
| mangueira | 683 | 6,7 | 343,8 | 0,725 |
| arvore_seca | 325 | 3,2 | 14,8 | 0,766 |
| bananeira | 190 | 1,9 | 33,9 | 0,732 |
| jabuticabeira | 140 | 1,4 | 9,9 | 0,656 |

Comparação: o run com Watershed (`runs/pipeline_cidade_20261103_alpha/`) deu 14.440 árvores. Teste e2e CC: 0,72 instância por árvore anotada (Watershed: 0,86), portanto o total da cidade é provavelmente subestimado.

## Revisão 2026-11-04

### Limiar escolhido na validação para as configurações herdadas (`runs/limiar_val_herdados_20261104/`)
Script `reavaliar_limiar_herdados.py` (mesma varredura 0,10–0,90 e mesmo `freeze_validation_threshold` da configuração principal; reproduziu o limiar 0,65 da principal). Teste avaliado uma vez com o limiar congelado.

| Configuração | Limiar | IoU val | IoU teste [IC95] | F1 | P | R |
|---|---|---|---|---|---|---|
| softAug4 (principal, U-Net ResNet-34) | 0,65 | 0,726 | 0,723 [0,691; 0,756] | 0,839 | 0,805 | 0,876 |
| 512semAug | 0,30 | 0,728 | 0,737 [0,703; 0,770] | 0,849 | 0,812 | 0,889 |
| _26 | 0,55 | 0,714 | 0,711 [0,677; 0,742] | 0,831 | 0,794 | 0,873 |
| 512semAug2 (treino_…) | 0,20 | 0,693 | 0,703 [0,667; 0,736] | 0,826 | 0,784 | 0,871 |
| base | 0,45 | 0,694 | 0,695 [0,659; 0,728] | 0,820 | 0,786 | 0,857 |
| 512softAug2 (treino_…) | 0,30 | 0,678 | 0,681 [0,638; 0,723] | 0,811 | 0,805 | 0,817 |

`20261101_185936_512semAug2` excluído (treino interrompido). Diferença pareada semAug − principal: val +0,002 [−0,016; 0,019], teste +0,014 [−0,005; 0,034] → principal mantida.

### Individualização nas máscaras preditas da validação (`runs/watershed_hmax_predita_val_20261104/`)
CC a_min 3000: F1 0,657 | Watershed janela (melhor f=251, a=3000): 0,612 | h-maxima (melhor h=1,0 m, a=2000): 0,633; F1 do h-maxima cresce com h e tende ao das CC. CC mantidas.

### Bootstrap por tile (`runs/bootstrap_ic_20261104/ic_95.csv`, B=2000, seed 42)
Classificador anotados F1-macro 0,639 [0,521; 0,730]; anotados − aumento offline +0,036 [−0,060; 0,139].
Bananeira binário 0,915 [0,857; 0,967] vs multiclasse 0,863 [0,788; 0,933]; dif. +0,052 [−0,001; 0,111].
Detecção e2e CC 0,631 [0,580; 0,685], WS 0,626 [0,579; 0,678]; dif. +0,005 [−0,034; 0,043]; instâncias/árvore CC 0,720 [0,655; 0,790].
Classificação e2e (CC) F1-macro 0,514 [0,401; 0,690]; acurácia 0,806 [0,753; 0,853].

### Classificador binário bananeira (`runs/classif_20261103_115814_efficientnet_binaria_bananeira/`)
Val F1-macro 0,865 (época 6). Teste (bananeira um-contra-todos): binário P 0,935 R 0,896 F1 0,915 AUC 0,964 AP 0,881 (TP 43 FP 3 FN 5); multiclasse P 0,815 R 0,917 F1 0,863 AUC 0,963 AP 0,856 (TP 44 FP 10 FN 4).

### Cidade com nova regra de união (`runs/pipeline_cidade_cc_uniao_20261104/`)
Regra `uma_parte_por_tile` (tree_georef.merge_border_crowns). Refeita só a união, a partir de `partes/` do run `pipeline_cidade_cc_20261103` (hardlinks). 19.982 instâncias → 11.077 árvores (3.206 unidas; maior grupo 479 partes, antes 803). Regra antiga: 10.251.

| Classe | Árvores | % | Área média (m²) | Confiança |
|---|---:|---:|---:|---:|
| sem identificação | 8747 | 79,0 | 80,7 | 0,869 |
| palmeira | 806 | 7,3 | 22,8 | 0,765 |
| mangueira | 795 | 7,2 | 292,8 | 0,732 |
| árvore seca | 362 | 3,3 | 14,4 | 0,770 |
| bananeira | 215 | 1,9 | 32,9 | 0,724 |
| jabuticabeira | 152 | 1,4 | 10,0 | 0,645 |

175 objetos > 500 m² somam 61% da área de copa (maior ≈ 9,2 ha). Teste: 301 instâncias → 225 árvores (regra antiga: 221).

### Amostra de verificação (`runs/amostra_verificacao_cidade_20261104/`)
100 árvores (20 sem identificação + 16 × 5 classes), fora dos tiles anotados, semente 42, com `peso_estrato`. Resultado: pendente (conferência manual do autor).

## Revisão 2026-11-05
- **Binários palmeira e mangueira:** treinados como a bananeira. Nenhum binário supera o multiclasse de forma distinguível do acaso.
  - O efeito é deslocar o ponto de decisão.
  - O multiclasse com limiar por espécie escolhido na validação reproduz o mesmo efeito; na mangueira, os falsos positivos caem de 33 para 13.
  - Detalhes em `runs/analise_binarios_20261105/RELATORIO.md`.
- **OSM, servidor de testes:** limiar de árvore `seg_prob_mean` ≥ 0,94, definitivo por decisão do autor (precisão no teste 0,988; mantém 31% das árvores). Só a bananeira é liberada para receber espécie.
  - Na cidade: 755 de 11.077 árvores selecionadas, com viés a favor da mangueira.
  - Enviadas ao servidor de testes em 2026-11-05 (changesets 719860, 719867, 719868; 755 nós). Depois, no changeset 720132, as bananeiras foram corrigidas para `natural=plant` + `genus=Musa` (8) ou apagadas (4, sem espécie). Restam 743 `natural=tree` e 8 `natural=plant`. Detalhes em `runs/osm_envio_simulacao_v2_20261105/RELATORIO_OSM.md`.
