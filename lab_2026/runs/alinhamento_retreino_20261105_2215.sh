#!/bin/bash
# Retreino das 6 configuracoes do benchmark de segmentacao com o crop sincronizado (branch fix/aligned-augmentation).
# Mesmos scripts, split, seeds e hiperparametros; so muda o alinhamento imagem-mascara. Novos runs: runs/alinhado_20261105_2215_<cfg>.
set -u
cd /home/paulomaciel/proj/lab_2026
P=/home/paulomaciel/anaconda3/envs/siace/bin/python
IMG=/home/paulomaciel/proj/imagens; MSK=/home/paulomaciel/proj/masks
COMMIT=$(git rev-parse HEAD)
run() {  # cfg comando...
  local id=alinhado_20261105_2215_$1; shift
  if [ -e runs/$id ]; then echo "$id ja existe, abortando"; exit 1; fi
  mkdir -p runs/$id
  { echo "[train] command=$*"; echo "[train] git_commit=$COMMIT"; echo "[train] started_at=$(date -u +%FT%T+00:00)"; } > runs/$id/log.txt
  ALIGNED_AUG_LOG=runs/$id/aug_params.csv "$@" --images-dir $IMG --masks-dir $MSK --run-root runs --run-id $id >> runs/$id/log.txt 2>&1
  local rc=$?
  echo "[train] ended_at=$(date -u +%FT%T+00:00) rc=$rc" >> runs/$id/log.txt
  echo "[$(date +%T)] $id rc=$rc"
}
run softAug4   $P -u tree_segmentation_resnet_26_512softAug4.py
run 512semAug  $P -u run_legacy_binary_training.py --script $PWD/tree_segmentation_resnet_26_512semAug.py
run 26         $P -u run_legacy_binary_training.py --script $PWD/tree_segmentation_resnet_26.py
run base       $P -u run_legacy_binary_training.py --script $PWD/tree_segmentation_resnet.py
run 512semAug2 $P -u run_legacy_binary_training.py --script $PWD/tree_segmentation_resnet_26_512semAug2.py
run 512softAug2 $P -u run_legacy_binary_training.py --script $PWD/tree_segmentation_resnet_26_512softAug2.py
echo "[$(date +%T)] retreino concluido"
