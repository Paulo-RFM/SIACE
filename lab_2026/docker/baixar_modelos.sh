#!/bin/bash
# Baixa os pesos dos modelos (não versionados no git) para as pastas runs/ esperadas pelo Dockerfile e
# confere o SHA-256 (o mesmo registrado em provenance.json das execuções avaliadas na monografia).
# Rodar a partir de lab_2026/. A URL base é a da release do GitHub onde os .pth forem publicados.
#   bash docker/baixar_modelos.sh [URL_BASE]
set -euo pipefail
URL=${1:-https://github.com/Paulo-RFM/SIACE/releases/download/modelos-v1}
baixar() {  # nome_na_release pasta sha256
  local nome=$1 dir=$2 sha=$3 dst=$2/checkpoint_best.pth
  mkdir -p "$dir"
  if [ -f "$dst" ] && echo "$sha  $dst" | sha256sum -c --status; then echo "ok (já existe): $dst"; return; fi
  [ -e "$dst" ] && { echo "$dst existe com outro conteúdo; não sobrescrevo"; exit 1; }
  curl -fL --retry 3 -o "$dst.part" "$URL/$nome"
  echo "$sha  $dst.part" | sha256sum -c --status || { echo "SHA-256 não confere: $nome"; rm -f "$dst.part"; exit 1; }
  mv "$dst.part" "$dst"; echo "baixado: $dst"
}
baixar segformer_3cm.pth runs/arq_segformer_mit_b1_20261110 5ae03d2fdbbfda404aa4a9dd57de460cad901c8ce90e2e896f1b8017cc5b45b5
baixar segformer_8cm.pth runs/res_segformer_8cm_20261121 37933afdd21a33699b93a765bbd3d13f77a415964efb830604f01a6e2f862806
baixar efficientnet_b0_especies.pth runs/classif_20261102_163825_efficientnet_anotados e4fd4f7790120e5af845ba75d9d3ed1025d30f5b225568c6506c26d48225df07
