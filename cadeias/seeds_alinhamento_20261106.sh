#!/bin/bash
# Repeticao com sementes do modelo principal (softAug4): codigo antigo (crop desalinhado, copia da tag
# baseline_before_alignment_fix) x codigo corrigido, sementes 1, 2 e 3, intercalados (antigo s1, corrigido s1, ...).
# Mesmo split, dados e hiperparametros; so mudam a semente e o alinhamento. Depois avalia com avaliar_seeds.py.
set -u
cd /home/paulomaciel/proj/lab_2026
P=/home/paulomaciel/anaconda3/envs/siace/bin/python
IMG=/home/paulomaciel/proj/imagens; MSK=/home/paulomaciel/proj/masks
STAMP=20261106
COMMIT=$(git rev-parse HEAD)
run() {  # grupo semente script
  local id=seeds_$1_s$2_$STAMP
  if [ -e runs/$id ]; then echo "$id ja existe, pulando"; return; fi
  mkdir -p runs/$id
  { echo "[train] command=$P -u $3 --seed $2"; echo "[train] git_commit=$COMMIT"; echo "[train] started_at=$(date -u +%FT%T+00:00)"; } > runs/$id/log.txt
  ALIGNED_AUG_LOG=runs/$id/aug_params.csv $P -u $3 --seed $2 --images-dir $IMG --masks-dir $MSK --run-root runs --run-id $id >> runs/$id/log.txt 2>&1
  local rc=$?
  echo "[train] ended_at=$(date -u +%FT%T+00:00) rc=$rc" >> runs/$id/log.txt
  echo "[$(date +%T)] $id rc=$rc"
}
for s in 1 2 3; do
  run antigo    $s tree_segmentation_resnet_26_512softAug4_codigo_antigo.py
  run corrigido $s tree_segmentation_resnet_26_512softAug4.py
done
echo "[$(date +%T)] treinos concluidos; avaliando"
$P -u avaliar_seeds.py --stamp $STAMP --seeds 1,2,3 --saida runs/seeds_alinhamento_comparacao_$STAMP
echo "[$(date +%T)] fim rc=$?"
