#!/bin/bash
# Repeticao com sementes da EfficientNet-B0 (criterio em runs/classif_seeds_efficientnet_20261120/criterio.md)
set -u
cd /home/paulomaciel/proj/lab_2026
P=/home/paulomaciel/anaconda3/envs/siace/bin/python
BASE="--manifest /home/paulomaciel/proj/species_dataset_v2/species_manifest.csv --split /home/paulomaciel/proj/unet_spatial_split/tiles_split.csv --species-names /home/paulomaciel/proj/lab_2026/species_names.json --runs-dir /home/paulomaciel/proj/lab_2026/runs"
AUG="--augmented-manifest /home/paulomaciel/proj/lab_2026/runs/classif_20261102_163825_efficientnet_aug_treino/train_augmented/train_augmented.csv"
git rev-parse HEAD > runs/classif_seeds_efficientnet_20261120/git_commit.txt
sha256sum train_species_efficientnet.py species_data.py species_metrics.py > runs/classif_seeds_efficientnet_20261120/scripts_sha256.txt
run() {  # variante semente
  local id=classif_seeds_20261120_efficientnet_$1_s$2
  if [ -e runs/$id/test_metrics_global.json ]; then echo "$id ja concluido, pulando"; return; fi
  local extra=""; [ "$1" = "aug_treino" ] && extra="$AUG"
  echo "[$(date +%T)] inicio $id"
  $P -u train_species_efficientnet.py $BASE $extra --seed $2 --run-id $id > runs/classif_seeds_efficientnet_20261120/$id.out 2>&1
  echo "[$(date +%T)] fim $id rc=$?"
}
for s in 1 2 3; do run anotados $s; run aug_treino $s; done
echo "[$(date +%T)] cadeia concluida"
