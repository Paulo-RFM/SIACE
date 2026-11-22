# Watershed calibrado x componentes conexas (2026-11-03) — somente VALIDAÇÃO

Pergunta: com peak_footprint=251 px (metade do tile), o watershed ainda separa copas, ou equivale a rotular
manchas conectadas? Mesma limpeza morfológica e mesmo min_area=1000 nos dois métodos.
Máscaras de REFERÊNCIA (`/home/paulomaciel/proj/masks/`), 203 tiles de validação, casamento húngaro IoU >= 0,5.
O teste NÃO foi usado.

| Método | Instâncias | TP | Precisão | Recall | F1 |
|---|---|---|---|---|---|
| Componentes conexas | 328 | 320 | 0,976 | 0,802 | 0,880 |
| Watershed (fp=251, min_area=1000) | 387 | 327 | 0,845 | 0,820 | 0,832 |

Em 134 de 203 tiles os dois métodos dão o mesmo número de instâncias. As 59 instâncias extras do watershed
renderam só 7 acertos a mais. Falta repetir com as máscaras PREDITAS pelo P1 na validação (precisa de GPU).
Script: `cc_baseline_val.py` (rodar a partir de lab_2026/).

## Máscaras PREDITAS pelo P1 (limiar 0,65), validação — `comparar_predita_val.py`

Máscaras geradas por `pipeline_cidade.py --partition val --save-masks` em `pipeline_val_predita/`.

| Método | min_area | Instâncias | Precisão | Recall | F1 |
|---|---|---|---|---|---|
| Componentes conexas | 1000 | 365 | 0,671 | 0,614 | 0,641 |
| Componentes conexas | 2000 | 348 | 0,704 | 0,614 | 0,656 |
| Watershed fp=251 (congelado) | 1000 | 404 | 0,599 | 0,607 | 0,603 |
| Watershed fp=121 | 1000 | 495 | 0,527 | 0,654 | 0,584 |

Conclusão: também na máscara predita as componentes conexas superam o watershed. O pipeline de
teste e da cidade continua com o watershed congelado (nada foi reavaliado no teste).
