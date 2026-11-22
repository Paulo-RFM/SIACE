#!/usr/bin/env bash
# Espera a suite de classificacao terminar e encadeia, usando a GPU sozinho:
#   1) pipeline nos 204 tiles de TESTE (com mascaras salvas)
#   2) avaliacao de ponta a ponta no teste (deteccao por arvore + classe), uma unica vez
#   3) pipeline na CIDADE inteira (tiles512), retomavel com --resume
# O classificador e escolhido pelo MAIOR F1-macro de VALIDACAO entre os runs completos de
# EfficientNet (nunca pelo teste).
#
# Uso (a partir de ~/proj/lab_2026, ambiente siace ativo ou nao):
#   nohup bash run_pos_suite.sh > ~/proj/pos_suite.out 2>&1 &
# Opcional: bash run_pos_suite.sh <PID_DA_SUITE>
set -euo pipefail
cd /home/paulomaciel/proj/lab_2026
PY=/home/paulomaciel/anaconda3/envs/siace/bin/python
CITY_TILES=/home/paulomaciel/Datasets/drone_rp_v01/tiles512
TS=$(date +%Y%m%d_%H%M%S)

SUITE_PID=${1:-$(pgrep -f "run_classification_suite.py" | head -1 || true)}
if [[ -n "${SUITE_PID}" ]]; then
  echo "[$(date +%H:%M:%S)] aguardando a suite (PID ${SUITE_PID}) terminar..."
  while kill -0 "${SUITE_PID}" 2>/dev/null; do sleep 60; done
fi
echo "[$(date +%H:%M:%S)] suite finalizada (ou nao estava rodando)."

CLS_RUN=$(${PY} - <<'EOF'
import json, glob, sys
best = None
for cfg in glob.glob("runs/classif_*_efficientnet_*/config.json"):
    c = json.load(open(cfg))
    if c.get("smoke") or "results" not in c or c.get("best_val_f1_macro") is None:
        continue
    if best is None or c["best_val_f1_macro"] > best[0]:
        best = (c["best_val_f1_macro"], cfg.rsplit("/", 1)[0])
if best is None:
    sys.exit("nenhum run completo de EfficientNet encontrado")
print(best[1])
EOF
)
echo "[$(date +%H:%M:%S)] classificador escolhido (maior F1-macro de validacao): ${CLS_RUN}"
grep -o '"best_val_f1_macro": [0-9.]*' "${CLS_RUN}/config.json" || true

TEST_DIR=runs/pipeline_test_${TS}
echo "[$(date +%H:%M:%S)] 1) pipeline nos tiles de teste -> ${TEST_DIR}"
${PY} -u pipeline_cidade.py --tiles-dir /home/paulomaciel/proj/imagens --partition test \
  --cls-run "${CLS_RUN}" --output-dir "${TEST_DIR}" --save-masks

echo "[$(date +%H:%M:%S)] 2) avaliacao de ponta a ponta no teste"
${PY} -u evaluate_pipeline_test.py --pipeline-dir "${TEST_DIR}"

CITY_DIR=runs/pipeline_cidade_${TS}
echo "[$(date +%H:%M:%S)] 3) pipeline na cidade -> ${CITY_DIR}"
${PY} -u pipeline_cidade.py --tiles-dir "${CITY_TILES}" --cls-run "${CLS_RUN}" \
  --output-dir "${CITY_DIR}" --resume

echo "[$(date +%H:%M:%S)] concluido. Teste: ${TEST_DIR} | Cidade: ${CITY_DIR}"
