#!/bin/bash
# Continuacao de runs/refazer_segformer_20261111.sh, que parou na calibracao (rc=3: o watershed h-maxima superou
# as componentes conexas na validacao com as mascaras da SegFormer). Decisao do autor (2026-11-11): adotar o
# h-maxima (h = 0,75 m, min_area 3000; runs/individualizacao_val_segformer_20261111/frozen_instance_params_hmax.json).
# Teste avaliado uma vez por metodo congelado: h-maxima (sistema), componentes conexas e watershed por janela.
# Rodar com: nohup bash runs/refazer_segformer_hmax_20261111.sh > runs/refazer_segformer_hmax_20261111.out 2>&1 &
set -u
cd /home/paulomaciel/proj/lab_2026
P=/home/paulomaciel/anaconda3/envs/siace/bin/python
IMG=/home/paulomaciel/proj/imagens
CIDADE=/home/paulomaciel/Datasets/drone_rp_v01/tiles512
SEG=runs/arq_segformer_mit_b1_20261110
ARQ="--seg-arch Segformer --seg-encoder mit_b1"
CLS=runs/classif_20261102_163825_efficientnet_anotados
PRED_TESTE=runs/arquiteturas_comparacao_20261110/arq_segformer_mit_b1_20261110/test_eval/predictions_test
HMAX=runs/individualizacao_val_segformer_20261111/frozen_instance_params_hmax.json
CC=runs/individualizacao_val_segformer_20261111/frozen_instance_params.json
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
etapa val_hmax $P -u pipeline_cidade.py --tiles-dir $IMG --partition val --seg-run $SEG $ARQ --cls-run $CLS \
  --ws-params $HMAX --output-dir runs/pipeline_val_hmax_$D --save-masks || exit 1
etapa teste_hmax $P -u pipeline_cidade.py --tiles-dir $IMG --partition test --seg-run $SEG $ARQ --cls-run $CLS \
  --ws-params $HMAX --output-dir runs/pipeline_test_hmax_$D --save-masks || exit 1
etapa teste_hmax_aval $P -u evaluate_pipeline_test.py --pipeline-dir runs/pipeline_test_hmax_$D --p1-predictions $PRED_TESTE || exit 1
etapa teste_cc $P -u pipeline_cidade.py --tiles-dir $IMG --partition test --seg-run $SEG $ARQ --cls-run $CLS \
  --ws-params $CC --output-dir runs/pipeline_test_$D --save-masks || exit 1
etapa teste_cc_aval $P -u evaluate_pipeline_test.py --pipeline-dir runs/pipeline_test_$D --p1-predictions $PRED_TESTE || exit 1
etapa teste_ws $P -u pipeline_cidade.py --tiles-dir $IMG --partition test --seg-run $SEG $ARQ --cls-run $CLS \
  --ws-params runs/watershed_ref_calib_20261102/frozen_watershed_params.json \
  --output-dir runs/pipeline_test_ws_$D --save-masks || exit 1
etapa teste_ws_aval $P -u evaluate_pipeline_test.py --pipeline-dir runs/pipeline_test_ws_$D --p1-predictions $PRED_TESTE || exit 1
etapa osm $P -u osm_calibrar_limiares.py --val-run runs/pipeline_val_hmax_$D --test-run runs/pipeline_test_hmax_$D \
  --saida runs/osm_limiares_$D || exit 1
etapa cidade $P -u pipeline_cidade.py --tiles-dir $CIDADE --seg-run $SEG $ARQ --cls-run $CLS --ws-params $HMAX \
  --output-dir runs/pipeline_cidade_$D --resume || exit 1
log "cadeia concluida"
