# Métricas dos modelos do SIACE

Gerado por `gerar_metricas_modelos.py` a partir dos arquivos dos runs; cada tabela cita o arquivo de origem. Divisão espacial por componentes conexas de tiles: 611 tiles de treino, 203 de validação e 204 de teste (1.018 tiles anotados, 2.113 copas). Limiar, checkpoint e parâmetros escolhidos na validação; teste avaliado uma vez por modelo. IC 95% por bootstrap por tile.

## Sistema final

| Etapa | Modelo | Run | Arquivo do modelo |
|---|---|---|---|
| Segmentação de copas | SegFormer MiT-B1, limiar 0,40 | `runs/arq_segformer_mit_b1_20261110` | `checkpoint_best.pth` |
| Individualização | Watershed h-maxima (h = 0,75 m, área mínima 3000 px) | `runs/individualizacao_val_segformer_20261111` | `frozen_instance_params_hmax.json` |
| Classificação de espécies | EfficientNet-B0 (recortes anotados) | `runs/classif_20261102_163825_efficientnet_anotados` | `checkpoint_best.pth` |
| Variante para imagens de 8 cm | SegFormer MiT-B1 treinada a 8 cm | `runs/res_segformer_8cm_20261121` | `checkpoint_best.pth` |

## 1. Segmentação binária: comparação de arquiteturas (semente 42, teste)

| Modelo | Parâm. | Limiar | IoU val. | IoU teste [IC 95%] | F1 | Precisão | Recall | Dif. IoU vs U-Net [IC 95%] | Inferência (s/tile) |
|---|---|---|---|---|---|---|---|---|---|
| U-Net (ResNet-34) | 24.436.369 | 0.4 | 0.748 | 0.751 [0.717; 0.781] | 0.858 | 0.810 | 0.912 | — | 0.107 |
| U-Net++ (ResNet-34) | 26.078.609 | 0.45 | 0.726 | 0.723 [0.689; 0.756] | 0.839 | 0.803 | 0.880 | -0.028 [-0.047; -0.008] | 0.146 |
| DeepLabV3+ (ResNet-34) | 22.437.457 | 0.55 | 0.740 | 0.744 [0.707; 0.777] | 0.853 | 0.806 | 0.906 | -0.007 [-0.025; +0.009] | 0.109 |
| SegFormer (MiT-B1) | 13.677.505 | 0.4 | 0.765 | 0.773 [0.739; 0.805] | 0.872 | 0.847 | 0.898 | +0.022 [+0.004; +0.040] | 0.123 |

U-Net (ResNet-34) é o modelo experimentado de referência; a SegFormer MiT-B1 é o modelo do sistema.

Fonte: `runs/arquiteturas_comparacao_20261110/resultados.csv`, `runs/arquiteturas_comparacao_20261110/ic_95.csv`

## 2. SegFormer MiT-B1 x U-Net ResNet-34 em quatro sementes

| Modelo | Semente | Limiar | IoU val. | IoU teste | F1 | Precisão | Recall |
|---|---|---|---|---|---|---|---|
| U-Net | 42 | 0.4 | 0.748 | 0.751 | 0.858 | 0.810 | 0.912 |
| U-Net | 1 | 0.5 | 0.756 | 0.755 | 0.860 | 0.810 | 0.917 |
| U-Net | 2 | 0.5 | 0.732 | 0.732 | 0.845 | 0.797 | 0.899 |
| U-Net | 3 | 0.4 | 0.730 | 0.740 | 0.851 | 0.812 | 0.893 |
| SegFormer | 42 | 0.4 | 0.765 | 0.773 | 0.872 | 0.847 | 0.898 |
| SegFormer | 1 | 0.55 | 0.758 | 0.743 | 0.852 | 0.820 | 0.887 |
| SegFormer | 2 | 0.7 | 0.766 | 0.771 | 0.871 | 0.846 | 0.897 |
| SegFormer | 3 | 0.5 | 0.770 | 0.757 | 0.862 | 0.845 | 0.879 |



| Métrica | U-Net (média ± dp) | SegFormer (média ± dp) | Diferença | p (permutação) | Sementes com SegFormer melhor |
|---|---|---|---|---|---|
| iou_val | 0.742 ± 0.013 | 0.765 ± 0.005 | +0.023 | 0.029 | 4/4 |
| iou | 0.744 ± 0.011 | 0.761 ± 0.014 | +0.017 | 0.114 | 3/4 |
| dice_f1 | 0.853 ± 0.007 | 0.864 ± 0.009 | +0.011 | 0.114 | 3/4 |
| precision | 0.807 ± 0.007 | 0.840 ± 0.013 | +0.032 | 0.029 | 4/4 |
| recall | 0.905 ± 0.011 | 0.890 ± 0.009 | -0.015 | 0.114 | 0/4 |


Critério de troca, fixado antes dos treinos: IoU médio de validação **e** de teste maiores nas quatro sementes. O teste, portanto, participou da escolha; a validação sozinha (4/4 sementes) levaria à mesma decisão.

```
sementes SegFormer completas: True
dif. media IoU validacao (SegFormer - U-Net): +0.0231
dif. media IoU teste (SegFormer - U-Net): +0.0166
criterio (ambas > 0, 4 sementes): SegFormer substitui a U-Net
```

Fonte: `runs/seeds_segformer_comparacao_20261110/resultados_por_run.csv`, `runs/seeds_segformer_comparacao_20261110/resumo.csv`, `runs/seeds_segformer_comparacao_20261110/decisao.txt`

## 3. Resolução da imagem (SegFormer treinada e testada a 3, 8 e 15 cm)

| Comparação | Métrica | Estimativa [IC 95%] |
|---|---|---|
| SegFormer 3 cm (sistema) no teste 3 cm | IoU teste | 0.773 [0.739; 0.805] |
| SegFormer 3 cm (sistema) no teste 8 cm | IoU teste | 0.772 [0.743; 0.799] |
| SegFormer 3 cm (sistema) no teste 15 cm | IoU teste | 0.733 [0.702; 0.764] |
| SegFormer 8 cm no teste 3 cm | IoU teste | 0.757 [0.724; 0.788] |
| SegFormer 8 cm no teste 8 cm | IoU teste | 0.772 [0.740; 0.802] |
| SegFormer 8 cm no teste 15 cm | IoU teste | 0.770 [0.741; 0.799] |
| SegFormer 15 cm no teste 3 cm | IoU teste | 0.718 [0.676; 0.757] |
| SegFormer 15 cm no teste 8 cm | IoU teste | 0.740 [0.705; 0.773] |
| SegFormer 15 cm no teste 15 cm | IoU teste | 0.750 [0.720; 0.779] |
| SegFormer 8 cm - SegFormer 3 cm (sistema), teste 3 cm | dif. IoU (pareada) | -0.016 [-0.033; +0.001] |
| SegFormer 8 cm - SegFormer 3 cm (sistema), teste 8 cm | dif. IoU (pareada) | +0.000 [-0.015; +0.014] |
| SegFormer 8 cm - SegFormer 3 cm (sistema), teste 15 cm | dif. IoU (pareada) | +0.037 [+0.017; +0.054] |
| SegFormer 15 cm - SegFormer 3 cm (sistema), teste 3 cm | dif. IoU (pareada) | -0.055 [-0.086; -0.025] |
| SegFormer 15 cm - SegFormer 3 cm (sistema), teste 8 cm | dif. IoU (pareada) | -0.033 [-0.053; -0.014] |
| SegFormer 15 cm - SegFormer 3 cm (sistema), teste 15 cm | dif. IoU (pareada) | +0.016 [-0.005; +0.035] |

Fonte: `runs/resolucao_comparacao_20261121/ic_95.csv`

## 4. Individualização das copas (validação, máscara predita pela SegFormer)

Escolhido: Watershed h-maxima, h = 0.75 m, área mínima 3000 px. F1 de detecção na validação (IoU ≥ 0,5): 0.679, contra 0.674 das componentes conexas (val (203 tiles, 399 arvores)).


Fonte: `runs/individualizacao_val_segformer_20261111/frozen_instance_params_hmax.json`, `runs/individualizacao_val_segformer_20261111/resultado.csv`

## 5. Detecção de árvores de ponta a ponta no teste (204 tiles, 420 árvores, IoU ≥ 0,5)

| Método | Instâncias | Acertos | Precisão | Recall | F1 | Instâncias por árvore |
|---|---|---|---|---|---|---|
| Componentes conexas | 308 | 241 | 0.782 | 0.574 | 0.662 | 0.73 |
| Watershed de janela fixa | 372 | 261 | 0.702 | 0.621 | 0.659 | 0.89 |
| Watershed h-maxima (sistema) | 355 | 260 | 0.732 | 0.619 | 0.671 | 0.85 |

Fonte: `runs/pipeline_test_segformer_20261111/avaliacao_teste/metricas_teste.json`, `runs/pipeline_test_ws_segformer_20261111/avaliacao_teste/metricas_teste.json`, `runs/pipeline_test_hmax_segformer_20261111/avaliacao_teste/metricas_teste.json`


Classificação nas 260 árvores detectadas corretamente pelo sistema: acurácia 82.3%, F1-macro 0.576, acurácia balanceada 0.624, kappa 0.582.

Fonte: `runs/pipeline_test_hmax_segformer_20261111/avaliacao_teste/metricas_teste.json`

## 6. Classificação de espécies: EfficientNet-B0 (teste, recortes anotados)

| N | Acurácia | Acurácia balanceada | F1-macro | F1 ponderado | Kappa | ECE | Melhor época |
|---|---|---|---|---|---|---|---|
| 418 | 0.835 | 0.741 | 0.639 | 0.853 | 0.644 | 0.067 | 17 |


Por classe:

```
                          precision    recall  f1-score   support

arvore_sem_identificacao     0.9640    0.8323    0.8933       322
                palmeira     0.6667    0.8000    0.7273        20
             arvore_seca     0.5000    0.6667    0.5714         3
               mangueira     0.3400    0.8947    0.4928        19
               bananeira     0.8148    0.9167    0.8627        48
           jabuticabeira     0.2500    0.3333    0.2857         6

                accuracy                         0.8349       418
               macro avg     0.5893    0.7406    0.6389       418
            weighted avg     0.8907    0.8349    0.8526       418
```

Fonte: `runs/classif_20261102_163825_efficientnet_anotados/test_metrics_global.json`, `runs/classif_20261102_163825_efficientnet_anotados/test_classification_report.txt`


Variação entre sementes (variante do sistema = `anotados`; `aug_treino` = variante com aumento offline):


| Variante | Semente | F1-macro val. | F1-macro teste | Acurácia bal. teste | Acurácia teste |
|---|---|---|---|---|---|
| anotados | 42 | 0.598 | 0.639 | 0.741 | 0.835 |
| anotados | 1 | 0.605 | 0.653 | 0.792 | 0.854 |
| anotados | 2 | 0.569 | 0.602 | 0.703 | 0.713 |
| anotados | 3 | 0.588 | 0.634 | 0.786 | 0.818 |
| aug_treino | 42 | 0.593 | 0.603 | 0.634 | 0.871 |
| aug_treino | 1 | 0.618 | 0.602 | 0.622 | 0.873 |
| aug_treino | 2 | 0.586 | 0.566 | 0.704 | 0.739 |
| aug_treino | 3 | 0.627 | 0.552 | 0.576 | 0.873 |


Média ± dp (4 sementes) e p de permutação:

```
{
  "val_f1": {
    "anotados": "0.590 ± 0.016",
    "aug_treino": "0.606 ± 0.020",
    "dif": 0.0158,
    "p": 0.257
  },
  "test_f1": {
    "anotados": "0.632 ± 0.022",
    "aug_treino": "0.580 ± 0.026",
    "dif": -0.0516,
    "p": 0.057
  },
  "test_bacc": {
    "anotados": "0.756 ± 0.042",
    "aug_treino": "0.634 ± 0.053",
    "dif": -0.1215,
    "p": 0.057
  },
  "test_acc": {
    "anotados": "0.805 ± 0.063",
    "aug_treino": "0.839 ± 0.067",
    "dif": 0.0341,
    "p": 0.457
  }
}
```

Fonte: `runs/classif_seeds_efficientnet_20261120/resultados_por_semente.csv`, `runs/classif_seeds_efficientnet_20261120/resumo.json`

## 7. Aplicação ao ortomosaico de Rio Paranaíba

- Tiles: 61426 (28775 sem imagem); tempo: 106 min.
- Árvores no ortomosaico inteiro (inclui área rural): 13491; no perímetro urbano (344.43 ha): 4854.
- Perímetro urbano por classe: arvore_sem_identificacao 3705, palmeira 468, mangueira 280, arvore_seca 116, bananeira 183, jabuticabeira 102.

Fonte: `runs/pipeline_cidade_segformer_20261111/summary.json`, `runs/pipeline_perimetro_segformer_20261121/contagem_perimetro.json`


Conferência visual de 100 árvores (amostra estratificada, um avaliador): detecção correta 73.4% (IC 95% 58.5% a 86.7%); classe correta 64.2% (IC 95% 48.9% a 78.4%).

Fonte: `runs/conferencia_100_arvores_20261121/metricas_conferencia.json`


Cobertura de copa no perímetro urbano: 6.0% (20.5 de 344.3 ha); 374 células de 100 m avaliadas; por faixa: 0-5% 228, 5-10% 91, 10-20% 34, 20-30% 12, 30-100% 9.
Validação no teste: cobertura predita 19.95% contra 18.82% da referência (superestimativa relativa de 6.0%, IC 95% 1.7% a 10.9%); erro absoluto médio por tile 3.3 p.p.; correlação por tile 0.93.

Fonte: `runs/cobertura_dossel_20261120/resumo.json`, `runs/cobertura_dossel_20261120/validacao_teste.json`


Filtro para o OpenStreetMap (limiar fixado na validação), no teste: 233 de 355 copas aprovadas, precisão 97.0% (sem filtro: 87.9%).

Fonte: `runs/osm_limiares_segformer_20261111/teste_resultado.json`

## 8. Teste em João Pessoa (ortoimagem de 8 cm, 3,61 km²)

| Modelo | Tiles | Árvores | Área de copa (ha) | Minutos de inferência |
|---|---|---|---|---|
| SegFormer 3 cm (sistema) | 14554 | 5110 | 43.83 | 31.7 |
| SegFormer 8 cm | 14554 | 5773 | 46.4 | 33.3 |
| SegFormer 15 cm | 14554 | 5217 | 52.55 | 32.9 |


Conferência visual de 100 árvores (amostra aleatória simples, semente 42): 92 sim, 7 não, 1 sem resposta; precisão de detecção 92.9% (IC 95% Wilson 86.1% a 96.5%); copas fundidas em 58 de 92.


Cobertura de copa da área imageada: 12.5% (43.79 de 350.62 ha); 378 células avaliadas, mediana 8.21%.


Fonte: pasta de João Pessoa (fora do repositório): `resultados/joao_pessoa/contagens_joao_pessoa.csv`, `conferencia_100/metricas_conferencia_jp.json`, `resultados/joao_pessoa/cobertura/resumo.json`. Os scripts estão em `joao_pessoa/`.
