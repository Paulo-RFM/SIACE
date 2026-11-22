#!/bin/bash
# Contagem de árvores só no perímetro urbano (Placemark.kml): os 13.360 tiles que tocam o perímetro
# (12.969 de tiles512 + 391 de novas_imagens, mesma lista da cobertura de dossel), com o sistema congelado
# (SegFormer MiT-B1 + h-maxima + EfficientNet-B0 semente 42), sem nenhuma alteração de parâmetro.
set -u
cd /home/paulomaciel/proj/lab_2026
P=/home/paulomaciel/anaconda3/envs/siace/bin/python
D=runs/pipeline_perimetro_segformer_20261121
echo "[$(date +%T)] inicio"
$P -u pipeline_cidade.py --tiles-dir $D/tiles_links --seg-run runs/arq_segformer_mit_b1_20261110 \
  --seg-arch Segformer --seg-encoder mit_b1 --cls-run runs/classif_20261102_163825_efficientnet_anotados \
  --ws-params runs/individualizacao_val_segformer_20261111/frozen_instance_params_hmax.json \
  --output-dir $D/saida --resume
echo "[$(date +%T)] fim rc=$?"
