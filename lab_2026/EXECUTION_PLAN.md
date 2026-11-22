# EXECUTION_PLAN — Classificação de espécies e fechamento da monografia

Estado em 2026-11-02 (2ª rodada):
- Auditoria concluída, manifesto validado e smoke da EfficientNet executado.
- Suíte de classificação criada e validada só por `--dry-run`, testes e avaliação de código. **Nenhum treino foi executado**, a pedido do autor.

Ambiente: `/home/paulomaciel/anaconda3/envs/siace/bin/python`, máquina Godel, GTX 1050 Ti com 4 GB.
Todos os comandos rodam a partir de `/home/paulomaciel/proj/lab_2026`.

## Decisões

| # | Item | Situação |
|---|---|---|
| D1 | Códigos `group_id` de `labels2/` iguais aos de `labels/` | **Resolvido**: confirmado pelo autor; manifesto com 2113 recortes (inclui 18 polígonos sem `group_id` como classe 1) |
| D2 | Nomes das classes | **Resolvido**: `species_names.json` (1 árvore sem identificação, 2 palmeira, 3 árvore seca, 4 mangueira, 5 bananeira, 6 jabuticabeira). Nomes científicos opcionais, a critério do autor |
| D3 | `labels2/OrtoRP-3cm_082_013 - Copia.json` | **Resolvido**: o autor decidiu excluí-lo (em 2026-11-02 o arquivo ainda estava em `labels2/`); já estava excluído do manifesto, então as contagens não mudam |
| D4 | Origem das máscaras | **Resolvido** (autor): `masks/` gerado por `make_mask.py` e `lab_2026/masks_distin2/` por `make_mask_distin.py`, a partir dos mesmos JSONs; imagens e máscaras casam 1:1 (verificado no pré-voo) |
| D5 | Descrição das diferenças entre os scripts de segmentação | **Aberto** (autor) |

## Suíte de classificação (`run_classification_suite.py`)

| Experimento | O que treina | Seleção do checkpoint | Teste |
|---|---|---|---|
| `efficientnet_anotados` | EfficientNet-B0 sobre os 2113 recortes anotados | máx. F1-macro val | dentro do treino, uma vez |
| `efficientnet_aug_treino` | Igual, + cópias aumentadas offline **só do treino** (300/classe) | idem | idem |
| `unet_mc_ce` | `resnet2.py`: U-Net multiclasse por pixel, CE, encoder sem pré-treino | mín. val_loss | `evaluate_multiclass_test.py` |
| `unet_mc_ce_dice` | `resnet3.py`: CE ponderada (pesos só do treino) + Dice, ImageNet | máx. mIoU val (fg presentes) | idem |
| `unet_mc_ce_dice_initP1` | `resnet3.py` inicializado com `runs/p1_20261101_140519/checkpoint_best.pth` (276/278 tensores; só a cabeça é nova) | idem | idem |
| *desativado* `efficientnet_pseudo` | — | — | sem CSV de pseudorrótulos válido |
| *desativado* `catboost_final.py` | — | — | outro problema (features `f0..f523`, alvo "attack") |

Saída de cada experimento em `runs/<suite_run_id>_<experimento>/`; resumo da suíte em `runs/<suite_run_id>_suite/summary.json`; índice em `lab_2026/index_classification.csv` (`build_classification_index.py`). O índice binário `index.csv` passa a ignorar esses runs.

## Passos e ordem recomendada

| Passo | Comando | Tempo estimado | Base da estimativa |
|---|---|---|---|
| 1. Conferir o plano | `python run_classification_suite.py --dry-run` | ~1 min | Executado: pré-voo OK |
| 2. Testes | `python -m unittest discover -s tests -v` | < 1 s | 30 testes OK |
| 3. Smoke de toda a suíte (recomendado) | `python run_classification_suite.py --smoke` | ~10–15 min | EfficientNet: 15 s/época de smoke medido; U-Net: 5 lotes + 10 tiles de teste por variante (não medido) |
| 4. Suíte completa | `nohup python run_classification_suite.py > classification_suite.out 2>&1 &` | **~3–5 h típico; ~9 h no pior caso** | EfficientNet só anotados: ~107 s/época, ≤ 40 épocas (≤ 72 min). Com aumento: ~2× (≤ 2,5 h). U-Net multiclasse: o P1 binário levou 93 s/época com a mesma arquitetura e lote; ≤ 50 épocas por variante (≤ 1,3–1,8 h cada), com early stopping (paciência 5) tendendo a encurtar |
| 5. Só um experimento | `python run_classification_suite.py --experiments unet_mc_ce_dice` | — | — |
| 6. Monografia | Preencher os `\pendente{}` de `resultados.tex` (§Desempenho do classificador) com `test_metrics_global.json`/`test_metrics_by_class.csv` (EfficientNet) e `test_eval/test_metrics.json`/`test_metrics_by_class.csv` (U-Net multiclasse); copiar `test_confusion_counts.png` para `matriz_confusao_especies.png` | 30–60 min | — |
| 7. (Opcional) Watershed sobre o P1 | Igual à 1ª rodada (gerar predições P1 com limiar 0,65 num diretório novo e depois `separate_tree_crowns_whatershed.py --mask-provenance predicao ...`) | ~10 min | — |
| 8. (Opcional) Pipeline integrado | Atualizar `pipeline_segment_classify_to_shapefile.py` l.44-45 e 63 para P1/0,65 e o melhor classificador; fornecer o ortomosaico | Autor | — |
| 9. Compilar o PDF | Overleaf ou outra máquina (sem LaTeX aqui) | Autor | — |

## Expectativas científicas (hipóteses, não resultados)

- "Árvore sem identificação" representa 77% do teste da EfficientNet e domina os pixels de árvore da U-Net multiclasse. Os números a reportar são F1-macro, acurácia balanceada e as métricas por classe, não a acurácia global.
- Árvore seca (3 no teste) e jabuticabeira (6) terão métricas por classe instáveis.
- As duas abordagens não são diretamente comparáveis. A EfficientNet é avaliada por recorte de árvore; a U-Net, por pixel e incluindo o fundo.
- A variante `initP1` testa se pesos aprendidos na segmentação binária ajudam a multiclasse. Só pode ser comparada com `unet_mc_ce_dice`, que tem a mesma perda e os mesmos dados.
