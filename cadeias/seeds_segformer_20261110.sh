#!/bin/bash
# Repeticao com sementes da SegFormer MiT-B1 (2026-11-10): sementes 1, 2 e 3, as mesmas da repeticao da U-Net
# (seeds_corrigido_s*_20261106). A semente 42 ja existe (arq_segformer_mit_b1_20261110).
# Mesmo protocolo (tree_segmentation_arquiteturas.py); depois avaliar_seeds_segformer.py compara com a U-Net.
# Rodar com: nohup bash runs/seeds_segformer_20261110.sh > runs/seeds_segformer_20261110.out 2>&1 &
set -u
cd /home/paulomaciel/proj/lab_2026
P=/home/paulomaciel/anaconda3/envs/siace/bin/python
IMG=/home/paulomaciel/proj/imagens; MSK=/home/paulomaciel/proj/masks
COMMIT=$(git rev-parse HEAD)
run() {  # semente
  local id=seeds_segformer_s$1_20261110
  if [ -e runs/$id ]; then echo "$id ja existe, pulando"; return; fi
  mkdir -p runs/$id
  { echo "[train] command=$P -u tree_segmentation_arquiteturas.py --arch Segformer --encoder mit_b1 --seed $1"; echo "[train] git_commit=$COMMIT";
    echo "[train] started_at=$(date -u +%FT%T+00:00)"; } > runs/$id/log.txt
  ALIGNED_AUG_LOG=runs/$id/aug_params.csv $P -u tree_segmentation_arquiteturas.py --arch Segformer --encoder mit_b1 --seed $1 \
    --images-dir $IMG --masks-dir $MSK --run-root runs --run-id $id >> runs/$id/log.txt 2>&1
  local rc=$?
  echo "[train] ended_at=$(date -u +%FT%T+00:00) rc=$rc" >> runs/$id/log.txt
  echo "[$(date +%T)] $id rc=$rc"
}
echo "[$(date +%T)] inicio"
for s in 1 2 3; do run $s; done
echo "[$(date +%T)] treinos concluidos; avaliando"
$P -u avaliar_seeds_segformer.py
echo "[$(date +%T)] fim rc=$?"
