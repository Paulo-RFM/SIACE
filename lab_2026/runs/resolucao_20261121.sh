#!/bin/bash
# Experimento de resolução (2026-11-21): SegFormer MiT-B1 treinada com imagens simuladas de 8 cm e de 15 cm
# (degradar_resolucao.py), com o mesmo script, semente, divisão espacial e máscaras da SegFormer do sistema
# (arq_segformer_mit_b1_20261110). Depois:
#   1. avalia os 3 modelos (atual 3 cm, 8 cm, 15 cm) no teste nas 3 resoluções (9 combinações; a do modelo
#      atual em 3 cm reaproveita a avaliação já feita, nada é reavaliado);
#   2. comparar_resolucao.py: tabelas, IC por bootstrap e gráficos;
#   3. aplica os 3 modelos à ortoimagem de João Pessoa (pipeline_cidade.py, h-maxima e EfficientNet do sistema)
#      e resume as contagens.
# Rodar com: nohup bash runs/resolucao_20261121.sh > runs/resolucao_20261121.out 2>&1 &
set -u
cd /home/paulomaciel/proj/lab_2026
P=/home/paulomaciel/anaconda3/envs/siace/bin/python
MSK=/home/paulomaciel/proj/masks
SPLIT=/home/paulomaciel/proj/unet_spatial_split
R=/home/paulomaciel/proj/resolucao_20261121
JP=/home/paulomaciel/proj_JoaoPessoa211126
OUT=runs/resolucao_comparacao_20261121
CLS=runs/classif_20261102_163825_efficientnet_anotados
HMAX=runs/individualizacao_val_segformer_20261111/frozen_instance_params_hmax.json
COMMIT=$(git rev-parse HEAD)
log() { echo "[$(date +%F' '%T)] $*"; }

treinar() {  # id pasta_imagens
  local id=$1 img=$2
  if [ -e runs/$id/checkpoint_best.pth ] && [ -e runs/$id/frozen_threshold.json ]; then log "$id ja treinado, pulando"; return 0; fi
  if [ -e runs/$id ]; then log "runs/$id existe incompleto; nao sobrescrevo"; return 1; fi
  mkdir -p runs/$id
  { echo "[train] command=$P -u tree_segmentation_arquiteturas.py --arch Segformer --encoder mit_b1 --images-dir $img"
    echo "[train] git_commit=$COMMIT"; echo "[train] started_at=$(date -u +%FT%T+00:00)"; } > runs/$id/log.txt
  ALIGNED_AUG_LOG=runs/$id/aug_params.csv $P -u tree_segmentation_arquiteturas.py --arch Segformer --encoder mit_b1 \
    --images-dir $img --masks-dir $MSK --run-root runs --run-id $id >> runs/$id/log.txt 2>&1
  local rc=$?
  echo "[train] ended_at=$(date -u +%FT%T+00:00) rc=$rc" >> runs/$id/log.txt
  log "$id rc=$rc"; return $rc
}

log "inicio"
treinar res_segformer_8cm_20261121 $R/imagens_8cm || exit 1
treinar res_segformer_15cm_20261121 $R/imagens_15cm || exit 1

for m in "atual:arq_segformer_mit_b1_20261110" "8cm:res_segformer_8cm_20261121" "15cm:res_segformer_15cm_20261121"; do
  mn=${m%%:*}; run=${m#*:}
  for t in "3cm:/home/paulomaciel/proj/imagens" "8cm:$R/imagens_8cm" "15cm:$R/imagens_15cm"; do
    tn=${t%%:*}; img=${t#*:}
    [ "$mn" = atual ] && [ "$tn" = 3cm ] && continue  # avaliação original reaproveitada
    ev=$OUT/eval/${mn}_em_${tn}
    if [ -e $ev/test_metrics.csv ]; then log "$ev ja existe"; continue; fi
    log "avaliando modelo $mn no teste $tn"
    $P -u evaluate_seg_arquiteturas_test.py --arch Segformer --encoder mit_b1 --images $img --masks $MSK \
      --split $SPLIT --checkpoint runs/$run/checkpoint_best.pth --threshold runs/$run/frozen_threshold.json \
      --output $ev || exit 1
  done
done
$P -u comparar_resolucao.py || exit 1

log "aguardando tiles de Joao Pessoa"
while ! grep -q "concluído" $JP/preparar_jp_3cm.out 2>/dev/null; do sleep 60; done
for m in "atual:arq_segformer_mit_b1_20261110" "8cm:res_segformer_8cm_20261121" "15cm:res_segformer_15cm_20261121"; do
  mn=${m%%:*}; run=${m#*:}
  log "Joao Pessoa com modelo $mn"
  SIACE_UTM_EPSG=31985 $P -u pipeline_cidade.py --tiles-dir $JP/tiles512_3cm_4326 --seg-run runs/$run \
    --seg-arch Segformer --seg-encoder mit_b1 --cls-run $CLS --ws-params $HMAX \
    --output-dir $OUT/joao_pessoa/$mn --resume || exit 1
done
$P -u comparar_resolucao.py --joao-pessoa || exit 1
log "fim"
