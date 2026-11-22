#!/bin/bash
# Refaz com a SegFormer MiT-B1 (runs/arq_segformer_mit_b1_20261110, semente 42, limiar 0,40 da validacao), que
# substitui a U-Net ResNet-34 no sistema (criterio das 4 sementes: runs/seeds_segformer_comparacao_20261110/decisao.txt),
# as mesmas etapas de runs/refazer_corrigido_20261106.sh 1-5: calibracao da individualizacao na validacao, ponta a
# ponta no teste (componentes conexas e watershed), filtros do OSM e cidade. Classificador inalterado.
# As U-Nets multiclasse (etapa 6 da cadeia anterior) nao dependem do modelo do sistema e nao sao refeitas.
# Rodar com: nohup bash runs/refazer_segformer_20261111.sh > runs/refazer_segformer_20261111.out 2>&1 &
set -u
cd /home/paulomaciel/proj/lab_2026
P=/home/paulomaciel/anaconda3/envs/siace/bin/python
IMG=/home/paulomaciel/proj/imagens
CIDADE=/home/paulomaciel/Datasets/drone_rp_v01/tiles512
SEG=runs/arq_segformer_mit_b1_20261110
ARQ="--seg-arch Segformer --seg-encoder mit_b1"
CLS=runs/classif_20261102_163825_efficientnet_anotados
PRED_TESTE=runs/arquiteturas_comparacao_20261110/arq_segformer_mit_b1_20261110/test_eval/predictions_test
D=segformer_20261111
log() { echo "[$(date +%T)] $*"; }
etapa() {
  local n=$1; shift
  if [ -e runs/refazer_$D/.ok_$n ]; then log "$n ja feito"; return 0; fi
  log "inicio $n"; "$@"; local rc=$?
  log "fim $n rc=$rc"
  [ $rc -eq 0 ] && touch runs/refazer_$D/.ok_$n
  return $rc
}
mkdir -p runs/refazer_$D
git rev-parse HEAD > runs/refazer_$D/git_commit.txt

etapa val_masks $P -u pipeline_cidade.py --tiles-dir $IMG --partition val --seg-run $SEG $ARQ --cls-run $CLS \
  --output-dir runs/pipeline_val_$D --save-masks || exit 1
etapa calib $P -u calibrar_individualizacao_val.py --masks runs/pipeline_val_$D/mascaras/binary \
  --seg-run $SEG --saida runs/individualizacao_val_$D
rc=$?; if [ $rc -ne 0 ]; then log "calibracao rc=$rc (3 = watershed superou componentes conexas): parando para revisao"; exit $rc; fi
WS=runs/individualizacao_val_$D/frozen_instance_params.json
AMIN=$($P -c "import json;print(json.load(open('$WS'))['min_area'])")
log "min_area escolhido na validacao: $AMIN"
VAL=runs/pipeline_val_$D
if [ "$AMIN" != "3000" ]; then
  VAL=runs/pipeline_val_amin${AMIN}_$D
  etapa val_amin $P -u pipeline_cidade.py --tiles-dir $IMG --partition val --seg-run $SEG $ARQ --cls-run $CLS \
    --ws-params $WS --output-dir $VAL --save-masks || exit 1
fi
etapa teste_cc $P -u pipeline_cidade.py --tiles-dir $IMG --partition test --seg-run $SEG $ARQ --cls-run $CLS \
  --ws-params $WS --output-dir runs/pipeline_test_$D --save-masks || exit 1
etapa teste_cc_aval $P -u evaluate_pipeline_test.py --pipeline-dir runs/pipeline_test_$D --p1-predictions $PRED_TESTE || exit 1
etapa teste_ws $P -u pipeline_cidade.py --tiles-dir $IMG --partition test --seg-run $SEG $ARQ --cls-run $CLS \
  --ws-params runs/watershed_ref_calib_20261102/frozen_watershed_params.json \
  --output-dir runs/pipeline_test_ws_$D --save-masks || exit 1
etapa teste_ws_aval $P -u evaluate_pipeline_test.py --pipeline-dir runs/pipeline_test_ws_$D --p1-predictions $PRED_TESTE || exit 1
etapa osm $P -u osm_calibrar_limiares.py --val-run $VAL --test-run runs/pipeline_test_$D \
  --saida runs/osm_limiares_$D || exit 1
etapa cidade $P -u pipeline_cidade.py --tiles-dir $CIDADE --seg-run $SEG $ARQ --cls-run $CLS --ws-params $WS \
  --output-dir runs/pipeline_cidade_$D --resume || exit 1
log "cadeia concluida"
