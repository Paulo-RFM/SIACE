#!/usr/bin/env python3
"""Sorteio de 100 árvores de João Pessoa para conferência visual no QGIS (2026-11-22).

Fonte: resultados do modelo do sistema (SegFormer MiT-B1 de 3 cm + h-maxima) na ortoimagem de João Pessoa,
runs/resolucao_comparacao_20261121/joao_pessoa/atual/ (cópia em ../arvore_joaopessoa.geojson).
Amostra aleatória simples de 100 das 5.110 árvores, semente 42, sem estratificação por classe: as classes de
espécie foram treinadas em Rio Paranaíba e não são conferidas aqui; mede-se só se a copa é de árvore real.
Saídas: amostra_100.gpkg (camadas "pontos", com os campos a preencher, e "copas") e as mesmas camadas em GeoJSON.
Uso (ambiente conda "geo"): python sortear.py
"""
import json
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()
HERE = Path(__file__).resolve().parent
SRC = Path("/home/paulomaciel/proj/lab_2026/runs/resolucao_comparacao_20261121/joao_pessoa/atual")
N, SEED = 100, 42

arv = json.loads((SRC / "arvores.geojson").read_text())
cop = json.loads((SRC / "copas.geojson").read_text())
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(len(arv["features"]), size=N, replace=False))
sel = [arv["features"][i] for i in idx]
ids = {f["properties"]["tree_id"]: k + 1 for k, f in enumerate(sel)}

def ponto(k, f):
    p = f["properties"]
    return {"type": "Feature", "geometry": f["geometry"], "properties": {
        "amostra_id": k + 1, "tree_id": p["tree_id"], "area_m2": round(p["area_m2"], 1), "n_parts": p["n_parts"],
        "e_arvore": "", "copas_fundidas": "", "obs": ""}}

pontos = {"type": "FeatureCollection", "crs": arv.get("crs"), "features": [ponto(k, f) for k, f in enumerate(sel)]}
copas = {"type": "FeatureCollection", "crs": cop.get("crs"), "features": [
    {"type": "Feature", "geometry": f["geometry"],
     "properties": {"amostra_id": ids[f["properties"]["tree_id"]], "tree_id": f["properties"]["tree_id"],
                    "area_m2": round(f["properties"]["area_m2"], 1)}}
    for f in cop["features"] if f["properties"]["tree_id"] in ids]}
assert len(copas["features"]) == N
(HERE / "amostra_100_pontos.geojson").write_text(json.dumps(pontos, ensure_ascii=False))
(HERE / "amostra_100_copas.geojson").write_text(json.dumps(copas, ensure_ascii=False))
gpkg = HERE / "amostra_100.gpkg"
if gpkg.exists():
    raise SystemExit(f"{gpkg} já existe; não sobrescrevo (pode conter respostas)")
gdal.VectorTranslate(str(gpkg), str(HERE / "amostra_100_pontos.geojson"), layerName="pontos", format="GPKG")
gdal.VectorTranslate(str(gpkg), str(HERE / "amostra_100_copas.geojson"), layerName="copas", format="GPKG",
                     accessMode="update")
print(f"{N} árvores sorteadas de {len(arv['features'])}; área mediana da copa na amostra "
      f"{np.median([f['properties']['area_m2'] for f in pontos['features']]):.1f} m²")
