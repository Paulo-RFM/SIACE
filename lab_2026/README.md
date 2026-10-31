# Projeto SIACE - Segmentação e classificação de espécies florestais

Este diretório reúne os experimentos, scripts auxiliares, métricas e materiais
do Trabalho de Conclusão de Curso sobre detecção de copas de árvores por
segmentação semântica e classificação de espécies.

O fluxo principal combina imagens aéreas multiespectrais, máscaras binárias ou
multiclasse, U-Net com encoders ResNet, EfficientNet-B0, dados sintéticos
rotulados pelo próprio modelo e um pipeline final de geoprocessamento.

## Como o projeto foi executado

### 1. Preparação das imagens e máscaras

As imagens e máscaras foram organizadas em diretórios com correspondência
entre os nomes dos arquivos. O processo histórico foi dividido em:

1. Corte das imagens aéreas em tiles.
2. Criação de máscaras binárias de copas.
3. Criação de máscaras distintas por instância ou por classe.
4. Criação de recortes das espécies a partir das coordenadas ou JSONs.
5. Balanceamento das classes por aumento de dados.

Scripts utilizados:

- `make_mask.py`: geração inicial de máscaras.
- `make_mask_distin.py`: geração de máscaras distintas.
- `generate_binary_masks.py`: geração de máscaras binárias.
- `separate_tree_crowns_whatershed.py`: separação de copas por watershed.
- `generate_crops_from_json.py`: geração de recortes a partir de JSON.
- `extract_zoomed_species_crops.py`: extração de recortes ampliados.
- `fix_species.py`: correção de rótulos ou nomes das espécies.
- `augment_rare_species.py`: aumento de dados das classes menos frequentes.
- `plot_species_distribution.py`: análise da distribuição das espécies.

### 2. Segmentação binária com U-Net

O experimento principal foi executado com U-Net e encoder ResNet-34. A loss
binária combinou BCE e Dice, com validação por IoU, acurácia, precisão, recall
e F1.

Scripts principais:

- `tree_segmentation_resnet.py`: treino básico da U-Net binária.
- `tree_segmentation_resnet_26.py`: variante histórica de 256 pixels.
- `tree_segmentation_resnet_26_semAug.py`: variante de 512 pixels sem aumento.
- `tree_segmentation_resnet_26_256softAug2.py`: experimento de 256 pixels.
- `tree_segmentation_resnet_26_256softAug3.py`: variante revisada com aumento
  moderado de dados.
- `tree_segmentation_resnet_26_512semAug.py`: experimento de 512 pixels.
- `tree_segmentation_resnet_26_512semAug2.py`: variante revisada sem aumento.
- `tree_segmentation_resnet_26_512softAug2.py`: aumento moderado em tiles de
  512 pixels.
- `tree_segmentation_resnet_26_512softAug4.py`: variante adicional de aumento
  em 512 pixels.
- `tree_segmentation_resnet_26_512Aug5.py`: experimento 512 pixels com aumento
  e pseudodataset.

Estes scripts foram executados diretamente em ambiente local, com GPU quando
disponível. As tentativas que apresentaram melhor desempenho por validação
salvaram checkpoints `.pth`; o histórico por época foi salvo em arquivos CSV.

Arquivos históricos já disponíveis:

- `unet_metrics.csv`
- `unet_metrics_256_softAug.csv`
- `unet_metrics_512_semAug.csv`
- `unet_metrics_512_softAug.csv`
- `unet_metrics_512_softAug2.csv`

### 3. Segmentação multiclasses

As variantes abaixo treinam máscaras com classes de árvores ou classes
derivadas:

- `resnet2.py`: segmento multiclasses com ajuste de pesos por classe.
- `resnet3.py`: variante multiclasses adicional.

Eles foram ajustados para registrar o histórico por época em CSV, incluindo
acurácia, precisão, recall, F1 e IoU.

### 4. Comparação de arquiteturas

- `banchmark_Unet.py`: compara encoders da U-Net e salva o resumo em CSV.
- `Unet_plus_banchmark.py`: compara encoders usando U-Net++ e registra o
  histórico por época e o resumo final.

Os resultados desses benchmarks devem ser reconciliados com os arquivos CSV e
com os checkpoints efetivamente gerados antes de serem apresentados na
monografia.

### 5. Classificação das espécies

O script `train_species_efficientnet.py` executa a classificação com
EfficientNet-B0 sobre seis classes. O melhor checkpoint é escolhido pelo F1
macro de validação.

A execução registra:

- histórico por época;
- métricas por classe;
- matriz de confusão;
- resumo JSON;
- melhor checkpoint.

Os arquivos são gravados no diretório configurado em `output_dir`, atualmente
`/home/paulomaciel/proj/species_model/`.

### 6. Pseudorótulos e aumento do conjunto

Os scripts seguintes foram usados para gerar pseudomáscaras ou conjuntos
balanceados:

- `make_pseudo_label_512.py`
- `generate_balanced_pseudo_dataset.py`
- `create_pseudo_dataset_classes.py`
- `create_pseudo_dataset61k_geral_classes_.py`

O pseudodataset foi usado como fonte adicional para algumas variantes de
treino. Não deve substituir a avaliação em dados reais e rotulados. O conjunto
de teste deve ser mantido isolado do treinamento e da geração de pseudorótulos.

### 7. Inferência geoespacial

O pipeline de predição segmenta as imagens, identifica instâncias de copas,
classifica as espécies e gera resultados vetoriais.

Scripts principais:

- `pipeline_predict_jsons.py`: predição com geração de JSON.
- `simple_pipeline_predict_masks.py`: geração simplificada de máscaras.
- `pipeline_segment_classify_to_shapefile.py`: segmentação, classificação e
  exportação para shapefile/geodatabase.
- `verificaGeo.py`: verificações geoespaciais auxiliares.
- `catboost_final.py`: classificação auxiliar com CatBoost.

## Orquestrador de treinamentos

O diretório possui um orquestrador opcional para executar ou listar os
experimentos conhecidos:

```bash
# Apenas listar os comandos que seriam executados
python run_training_suite.py --dry-run

# Executar o treino binário básico e o classificador
python run_training_suite.py unet-basic species-classification

# Executar todos os experimentos cadastrados
python run_training_suite.py
```

O orquestrador não confirma que todos os arquivos de dados e modelos estão
presentes. Os caminhos absolutos dos scripts devem ser conferidos antes da
execução.

## Métricas registradas

Para segmentação, as métricas esperadas são:

- perda;
- acurácia por pixel;
- precisão;
- recall;
- F1;
- IoU;
- contagens agregadas TP, FP, FN e TN.

Para classificação, são esperadas:

- acurácia;
- precisão, recall e F1 macro;
- métricas por classe;
- matriz de confusão;
- melhor checkpoint;
- resumo JSON.

As métricas agregadas de segmentação devem ser calculadas sobre contagens
acumuladas de pixels. Médias de métricas calculadas por batch podem produzir
valores incorretos para o experimento completo.

## Caminhos e ambiente

Diversos scripts usam caminhos absolutos da máquina original, por exemplo:

```text
/home/paulomaciel/proj/imagens/
/home/paulomaciel/proj/masks/
/home/paulomaciel/proj/species_crops_balanced/
```

Ao executar em outra máquina, altere os caminhos definidos na seção de
configuração de cada script. Prefira manter uma estrutura semelhante à abaixo:

```text
<projeto>/
├── imagens/
├── masks/
├── labels/
├── species_crops/
├── species_crops_balanced/
├── pseudo_dataset/
├── modelos/
└── resultados/
```

### Dependências principais

Instale as bibliotecas necessárias ao ambiente de execução:

```bash
pip install numpy pandas pillow scikit-learn matplotlib tqdm torch torchvision
pip install segmentation-models-pytorch efficientnet-pytorch albumentations
```

Dependências adicionais podem ser necessárias para o pipeline geoespacial,
como `geopandas`, `shapely`, `rasterio` e `pyproj`. Instale apenas os pacotes
necessários ao script utilizado.

## Organização dos resultados

- arquivos `.csv`: histórico ou resumo das métricas;
- arquivos `.json`: resumos, labels e metadados;
- arquivos `.pth`: pesos dos modelos treinados;
- arquivos de imagens: máscaras, gráficos e recortes;
- `LaTex/`: monografia, figuras e arquivos auxiliares;
- arquivos `.save`: cópias antigas de scripts; não devem ser usados como fonte
  principal do experimento.

## Reprodutibilidade

Antes de um novo ciclo de treino:

1. Prepare pares imagem-máscara consistentes.
2. Declare ou gere uma divisão por mosaico ou região geográfica.
3. Reserve um conjunto de teste fixo antes dos experimentos.
4. Confirme que não existem tiles ou imagens derivadas de um mesmo mosaico em
   diferentes divisões.
5. Execute apenas o aumento de dados sobre o conjunto de treino.
6. Mantenha validação e teste sem aumento.
7. Execute o script escolhido pelo orquestrador ou diretamente.
8. Confira o CSV, o checkpoint e o resumo gerados.
9. Compare novos modelos com o mesmo conjunto de teste da linha de base.

Os resultados históricos não devem ser misturados com novos resultados sem
documentar o conjunto de dados, os pesos utilizados, a divisão, as épocas e
o hardware.
