#!/bin/bash
# Cobertura de dossel da cidade: espera os treinos da EfficientNet liberarem a GPU (PID 29319) e roda na GPU.
set -u
cd /home/paulomaciel/proj/lab_2026
P=/home/paulomaciel/anaconda3/envs/siace/bin/python
while kill -0 29319 2>/dev/null; do sleep 60; done
echo "[$(date +%T)] GPU livre; inicio cidade"
$P -u cobertura_dossel.py cidade --device cuda --batch 8 || exit 1
$P -u cobertura_dossel.py mapa || exit 1
echo "[$(date +%T)] concluido"
