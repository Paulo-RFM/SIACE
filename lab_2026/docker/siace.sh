#!/bin/bash
# Ponto de entrada do contêiner SIACE.
#   siace processar <ortomosaico.tif> <pasta_saida> [--modelo 3cm|8cm] [--utm EPSG]
#   siace pipeline  <pasta_tiles> <pasta_saida> [--modelo 3cm|8cm] [--utm EPSG]   (tiles já prontos)
# Modelos: 3cm = sistema (SegFormer MiT-B1, ortomosaico de 3,19 cm); 8cm = variante treinada com imagens de 8 cm.
set -euo pipefail
APP=$(cd "$(dirname "$0")" && pwd)
ajuda() { sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; }
cmd=${1:-ajuda}; shift || true
[ "$cmd" = processar ] || [ "$cmd" = pipeline ] || { ajuda; exit 0; }
[ $# -ge 2 ] || { ajuda; exit 1; }
ENTRADA=$1; SAIDA=$2; shift 2
MODELO=3cm; UTM=""
while [ $# -gt 0 ]; do
  case $1 in
    --modelo) MODELO=$2; shift 2;;
    --utm) UTM=$2; shift 2;;
    *) echo "opção desconhecida: $1"; exit 1;;
  esac
done
case $MODELO in
  3cm) SEG=runs/arq_segformer_mit_b1_20261110;;
  8cm) SEG=runs/res_segformer_8cm_20261121;;
  *) echo "modelo deve ser 3cm ou 8cm"; exit 1;;
esac
mkdir -p "$SAIDA"
if [ "$cmd" = processar ]; then
  python3 "$APP/preparar_ortomosaico.py" "$ENTRADA" --saida "$SAIDA/preparo"
  TILES=$SAIDA/preparo/tiles
  [ -n "$UTM" ] || UTM=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['utm_epsg'])" "$SAIDA/preparo/preparo.json")
else
  TILES=$ENTRADA
  [ -n "$UTM" ] || UTM=31983
fi
echo "modelo $MODELO ($SEG); UTM EPSG:$UTM; tiles $TILES"
cd "$APP"
SIACE_UTM_EPSG=$UTM python3 -u pipeline_cidade.py --tiles-dir "$TILES" --output-dir "$SAIDA/resultado_$MODELO" \
  --seg-run "$SEG" --seg-arch Segformer --seg-encoder mit_b1 \
  --ws-params runs/individualizacao_val_segformer_20261111/frozen_instance_params_hmax.json \
  --cls-run runs/classif_20261102_163825_efficientnet_anotados --resume
echo "pronto: $SAIDA/resultado_$MODELO (arvores.geojson = pontos; copas.geojson = contornos)"
