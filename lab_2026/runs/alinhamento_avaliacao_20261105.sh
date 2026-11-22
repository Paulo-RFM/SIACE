#!/bin/bash
# Espera o retreino alinhado terminar e avalia: limiar na validacao, teste unico, comparacao pareada e e2e.
set -euo pipefail
cd /home/paulomaciel/proj/lab_2026
P=/home/paulomaciel/anaconda3/envs/siace/bin/python
while kill -0 "$(cat /home/paulomaciel/proj/alinhamento_retreino.pid)" 2>/dev/null; do sleep 60; done
echo "[$(date +%T)] retreino terminou"
$P avaliar_alinhamento.py --stamp 20261105_2215 --saida runs/alinhamento_comparacao_20261105
$P comparar_e2e_alinhamento.py --seg-run runs/alinhado_20261105_2215_softAug4 \
   --pred-novo runs/alinhamento_comparacao_20261105/softAug4/test_eval/predictions_test \
   --pipeline-dir runs/pipeline_test_alinhado_20261105 --saida runs/alinhamento_comparacao_20261105
echo "[$(date +%T)] avaliacao concluida"
