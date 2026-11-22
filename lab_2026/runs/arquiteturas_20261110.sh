#!/bin/bash
# Benchmark de arquiteturas de segmentacao (2026-11-10): U-Net++, DeepLabV3+ e SegFormer com o mesmo
# protocolo da U-Net ResNet-34 principal (tree_segmentation_arquiteturas.py; semente 42, lote 6, 512 px).
# Treinos em sequencia; depois avaliar_arquiteturas.py avalia o teste uma vez por modelo e resume.
# Rodar com: nohup bash runs/arquiteturas_20261110.sh > runs/arquiteturas_20261110.out 2>&1 &
set -u
cd /home/paulomaciel/proj/lab_2026
P=/home/paulomaciel/anaconda3/envs/siace/bin/python
IMG=/home/paulomaciel/proj/imagens; MSK=/home/paulomaciel/proj/masks
COMMIT=$(git rev-parse HEAD)
run() {  # id arquitetura codificador
  local id=$1
  if [ -e runs/$id ]; then echo "$id ja existe, pulando"; return; fi
  mkdir -p runs/$id
  { echo "[train] command=$P -u tree_segmentation_arquiteturas.py --arch $2 --encoder $3"; echo "[train] git_commit=$COMMIT";
    echo "[train] started_at=$(date -u +%FT%T+00:00)"; } > runs/$id/log.txt
  ALIGNED_AUG_LOG=runs/$id/aug_params.csv $P -u tree_segmentation_arquiteturas.py --arch $2 --encoder $3 \
    --images-dir $IMG --masks-dir $MSK --run-root runs --run-id $id >> runs/$id/log.txt 2>&1
  local rc=$?
  echo "[train] ended_at=$(date -u +%FT%T+00:00) rc=$rc" >> runs/$id/log.txt
  echo "[$(date +%T)] $id rc=$rc"
}
echo "[$(date +%T)] inicio"
run arq_deeplabv3plus_resnet34_20261110 DeepLabV3Plus resnet34
run arq_segformer_mit_b1_20261110 Segformer mit_b1
run arq_unetplusplus_resnet34_20261110 UnetPlusPlus resnet34
echo "[$(date +%T)] treinos concluidos; avaliando"
$P -u avaliar_arquiteturas.py
echo "[$(date +%T)] fim rc=$?"
