"""Sorteio de 100 arvores da cidade para verificacao manual no QGIS (P2 da revisao critica).

Fonte: runs/pipeline_cidade_corrigido_20261106/ (sistema final: U-Net ResNet-34 corrigida, componentes conexas).
Copia de runs/amostra_verificacao_cidade_20261104/sortear.py; so a fonte mudou.
* Exclui as arvores cujos tiles foram anotados (os 1.018 do treino/validacao/teste), para que a
  amostra seja independente dos dados usados no desenvolvimento.
* Estratificado por classe predita: 20 "sem identificacao" e 16 de cada uma das outras 5 classes.
  Assim cada classe tem amostra suficiente para estimar sua precisao; o total da cidade se estima
  reponderando cada estrato pelo seu tamanho (peso = N_classe / n_classe, coluna `peso_estrato`).
* Semente fixa (42). Rodar a partir de lab_2026/.
Saidas: amostra_100.csv (planilha para preencher) e amostra_100_pontos.geojson / amostra_100_copas.geojson.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CITY = HERE.parent / "pipeline_cidade_corrigido_20261106"
SPLIT = Path("/home/paulomaciel/proj/unet_spatial_split/tiles_split.csv")
N = {"arvore_sem_identificacao": 20, "palmeira": 16, "mangueira": 16, "arvore_seca": 16, "bananeira": 16,
     "jabuticabeira": 16}

arv = pd.read_csv(CITY / "arvores.csv")
anot = set(pd.read_csv(SPLIT).tile)
tiles_of = arv.parts.str.split(";").apply(lambda ps: {p.split("#")[0] for p in ps})
pool = arv[~tiles_of.apply(lambda ts: bool(ts & anot))]
rng = np.random.default_rng(42)
amostra = []
for cls, n in N.items():
    sub = pool[pool.pred_name == cls]
    pick = sub.iloc[np.sort(rng.choice(len(sub), size=n, replace=False))].copy()
    pick["estrato_N"] = int((arv.pred_name == cls).sum())
    pick["peso_estrato"] = pick["estrato_N"] / n
    amostra.append(pick)
a = pd.concat(amostra).reset_index(drop=True)
a.insert(0, "amostra_id", np.arange(1, len(a) + 1))
cols = ["amostra_id", "tree_id", "pred_name", "confidence", "area_m2", "n_parts", "tile", "parts", "lon", "lat",
        "estrato_N", "peso_estrato"]
out = a[cols].copy()
# colunas para preencher na verificacao
out["e_arvore"] = ""            # s / n (o ponto cai sobre uma copa de arvore real?)
out["classe_correta"] = ""      # s / n / ? (a classe predita e plausivel?)
out["classe_observada"] = ""    # nome da classe vista, se diferente
out["copas_fundidas"] = ""      # numero de arvores reais dentro da copa delimitada (1 = ok)
out["deslocamento_m"] = ""      # distancia (m) entre o ponto e o centro da copa na imagem de referencia
out["referencia"] = ""          # imagem usada para medir o deslocamento (ex.: Google Satellite, Esri, ...)
out["obs"] = ""
out.to_csv(HERE / "amostra_100.csv", index=False)

feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
          "properties": {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}}
         for _, r in out.iterrows()]
crs = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
(HERE / "amostra_100_pontos.geojson").write_text(json.dumps({"type": "FeatureCollection", "name": "amostra_100_pontos",
                                                             "crs": crs, "features": feats}), encoding="utf-8")
ids = dict(zip(out.tree_id, out.amostra_id))
copas = json.load(open(CITY / "copas.geojson"))
sel = []
for f in copas["features"]:
    t = f["properties"].get("tree_id")
    if t in ids:
        f["properties"]["amostra_id"] = int(ids[t])
        sel.append(f)
(HERE / "amostra_100_copas.geojson").write_text(json.dumps({"type": "FeatureCollection", "name": "amostra_100_copas",
                                                            "crs": crs, "features": sel}), encoding="utf-8")
print(f"pool sem tiles anotados: {len(pool)} de {len(arv)} arvores; amostra {len(out)}; copas {len(sel)}")
print(out.groupby("pred_name").size().to_string())
