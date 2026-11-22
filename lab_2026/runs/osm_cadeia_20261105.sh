#!/bin/bash
# Cadeia da tarefa OSM (2026-11-05). Espera a suite de treinos binarios liberar a GPU e entao:
#  1. pipeline (com seg_prob_mean) na validacao e no teste, salvando mascaras;
#  2. confere que o teste reproduz as copas e classes do run anterior (so acrescenta o campo novo);
#  3. calibra os limiares OSM na validacao e mede no teste (osm_calibrar_limiares.py);
#  4. pipeline na cidade inteira com seg_prob_mean;
#  5. simulacao do envio (sem --enviar: nada sai da maquina).
set -euo pipefail
cd /home/paulomaciel/proj/lab_2026
PY=/home/paulomaciel/anaconda3/envs/siace/bin/python
CLS=runs/classif_20261102_163825_efficientnet_anotados
while kill -0 "$(cat /home/paulomaciel/proj/binarias_palm_mang.pid)" 2>/dev/null; do sleep 30; done
echo "[$(date +%T)] GPU livre"
$PY pipeline_cidade.py --tiles-dir /home/paulomaciel/proj/imagens --partition val --cls-run $CLS \
    --output-dir runs/pipeline_val_segprob_20261105 --save-masks
$PY pipeline_cidade.py --tiles-dir /home/paulomaciel/proj/imagens --partition test --cls-run $CLS \
    --output-dir runs/pipeline_test_segprob_20261105 --save-masks
$PY - <<'EOF'
import pandas as pd
a = pd.read_csv("runs/pipeline_test_cc_20261103_223209/instancias.csv")
b = pd.read_csv("runs/pipeline_test_segprob_20261105/instancias.csv")
k = ["tile", "instance_id", "n_pixels", "pred_class"]
same = a[k].sort_values(k[:2]).reset_index(drop=True).equals(b[k].sort_values(k[:2]).reset_index(drop=True))
print("teste reproduz o run anterior (copas e classes):", same, len(a), len(b))
if not same:
    raise SystemExit("divergencia: verificar antes de seguir")
EOF
$PY osm_calibrar_limiares.py --val-run runs/pipeline_val_segprob_20261105 --test-run runs/pipeline_test_segprob_20261105
$PY pipeline_cidade.py --tiles-dir /home/paulomaciel/Datasets/drone_rp_v01/tiles512 --cls-run $CLS \
    --output-dir runs/pipeline_cidade_segprob_20261105 --resume
$PY osm_envio_dev.py --cidade-run runs/pipeline_cidade_segprob_20261105 \
    --limiares runs/osm_limiares_20261105/limiares_osm.json --saida runs/osm_envio_simulacao_20261105
echo "[$(date +%T)] cadeia concluida"
