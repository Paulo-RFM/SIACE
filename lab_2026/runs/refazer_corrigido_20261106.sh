#!/bin/bash
# Refaz, com a U-Net ResNet-34 CORRIGIDA (crop sincronizado; runs/alinhado_20261105_2215_softAug4, semente 42),
# todas as etapas que dependiam do modelo antigo: calibracao da individualizacao na validacao, ponta a ponta no
# teste (componentes conexas e watershed), aplicacao a cidade, calibracao dos filtros do OSM e as U-Nets
# multiclasse treinadas com resnet3.py (que tinha o mesmo defeito de crop). Nada antigo e sobrescrito.
set -u
cd /home/paulomaciel/proj/lab_2026
P=/home/paulomaciel/anaconda3/envs/siace/bin/python
IMG=/home/paulomaciel/proj/imagens
CIDADE=/home/paulomaciel/Datasets/drone_rp_v01/tiles512
SEG=runs/alinhado_20261105_2215_softAug4
CLS=runs/classif_20261102_163825_efficientnet_anotados
PRED_TESTE=runs/alinhamento_comparacao_20261105/softAug4/test_eval/predictions_test
D=20261106
log() { echo "[$(date +%T)] $*"; }
etapa() {  # nome comando...  (pula se a saida ja existe: marcador runs/.ok_<nome>)
  local n=$1; shift
  if [ -e runs/refazer_$D/.ok_$n ]; then log "$n ja feito"; return 0; fi
  log "inicio $n"; "$@"; local rc=$?
  log "fim $n rc=$rc"
  [ $rc -eq 0 ] && touch runs/refazer_$D/.ok_$n
  return $rc
}
mkdir -p runs/refazer_$D
git rev-parse HEAD > runs/refazer_$D/git_commit.txt

# 1. mascaras preditas da validacao
etapa val_masks $P -u pipeline_cidade.py --tiles-dir $IMG --partition val --seg-run $SEG --cls-run $CLS \
  --output-dir runs/pipeline_val_corrigido_$D --save-masks || exit 1
# 2. calibracao da individualizacao (so validacao); para se algum watershed superar as componentes conexas
etapa calib $P -u calibrar_individualizacao_val.py --masks runs/pipeline_val_corrigido_$D/mascaras/binary \
  --seg-run $SEG --saida runs/individualizacao_val_corrigido_$D
rc=$?; if [ $rc -ne 0 ]; then log "calibracao rc=$rc (3 = watershed superou componentes conexas): parando para revisao"; exit $rc; fi
WS=runs/individualizacao_val_corrigido_$D/frozen_instance_params.json
AMIN=$($P -c "import json;print(json.load(open('$WS'))['min_area'])")
log "min_area escolhido na validacao: $AMIN"
VAL=runs/pipeline_val_corrigido_$D
if [ "$AMIN" != "3000" ]; then
  VAL=runs/pipeline_val_corrigido_amin${AMIN}_$D
  etapa val_amin $P -u pipeline_cidade.py --tiles-dir $IMG --partition val --seg-run $SEG --cls-run $CLS \
    --ws-params $WS --output-dir $VAL --save-masks || exit 1
fi
# 3. ponta a ponta no teste: componentes conexas (sistema final) e watershed calibrado nas referencias
etapa teste_cc $P -u pipeline_cidade.py --tiles-dir $IMG --partition test --seg-run $SEG --cls-run $CLS \
  --ws-params $WS --output-dir runs/pipeline_test_corrigido_$D --save-masks || exit 1
etapa teste_cc_aval $P -u evaluate_pipeline_test.py --pipeline-dir runs/pipeline_test_corrigido_$D --p1-predictions $PRED_TESTE || exit 1
etapa teste_ws $P -u pipeline_cidade.py --tiles-dir $IMG --partition test --seg-run $SEG --cls-run $CLS \
  --ws-params runs/watershed_ref_calib_20261102/frozen_watershed_params.json \
  --output-dir runs/pipeline_test_ws_corrigido_$D --save-masks || exit 1
etapa teste_ws_aval $P -u evaluate_pipeline_test.py --pipeline-dir runs/pipeline_test_ws_corrigido_$D --p1-predictions $PRED_TESTE || exit 1
# 4. filtros do OSM: calibrados na validacao, medidos uma vez no teste
etapa osm $P -u osm_calibrar_limiares.py --val-run $VAL --test-run runs/pipeline_test_corrigido_$D \
  --saida runs/osm_limiares_corrigido_$D || exit 1
# 5. cidade inteira
etapa cidade $P -u pipeline_cidade.py --tiles-dir $CIDADE --seg-run $SEG --cls-run $CLS --ws-params $WS \
  --output-dir runs/pipeline_cidade_corrigido_$D || exit 1
# 6. U-Nets multiclasse com resnet3.py corrigido (CE+Dice; e CE+Dice inicializada com a U-Net binaria corrigida)
MC="--images-dir $IMG --masks-dir /home/paulomaciel/proj/lab_2026/masks_distin2 --run-root /home/paulomaciel/proj/lab_2026/runs --split /home/paulomaciel/proj/unet_spatial_split/tiles_split.csv --species-names /home/paulomaciel/proj/lab_2026/species_names.json"
for v in ce_dice ce_dice_initbin; do
  id=unet_mc_${v}_corrigido_$D
  extra=""; [ $v = ce_dice_initbin ] && extra="--init-checkpoint /home/paulomaciel/proj/lab_2026/$SEG/checkpoint_best.pth"
  etapa mc_$v $P -u run_legacy_multiclass_training.py --script /home/paulomaciel/proj/lab_2026/resnet3.py $MC --run-id $id $extra || continue
  etapa mc_${v}_teste $P -u evaluate_multiclass_test.py --images $IMG --masks /home/paulomaciel/proj/lab_2026/masks_distin2 \
    --split /home/paulomaciel/proj/unet_spatial_split/tiles_split.csv --checkpoint runs/$id/checkpoint_best.pth \
    --output runs/$id/test_eval --normalize imagenet --species-names /home/paulomaciel/proj/lab_2026/species_names.json
done
log "cadeia concluida"
