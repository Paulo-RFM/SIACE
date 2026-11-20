#!/usr/bin/env python3
"""Altura (MDS - MDT no ponto) no filtro do OSM (2026-11-18). Criterio: runs/filtro_altura_20261118/criterio.md (adendo).
Uso: python osm_filtro_altura.py   (validacao + teste uma vez; recusa se o teste ja tiver sido avaliado)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from filtro_altura import alturas

LAB = Path(__file__).resolve().parent
OSM = LAB / "runs" / "osm_limiares_segformer_20261111"
OUT = LAB / "runs" / "filtro_altura_20261118"
LIM = json.loads((OSM / "limiares_osm.json").read_text())
AMAX, META = LIM["area_max_m2"], LIM["precisao_arvore_alvo"]
SG = np.round(np.arange(0.50, 0.99, 0.01), 2)
HG = [None] + [round(0.25 * k, 2) for k in range(1, 13)]


def carregar(nome, H):
    d = pd.read_csv(OSM / nome)
    d["alt"] = [H.get((t, int(i)), np.nan) for t, i in zip(d.tile, d.instance_id)]
    return d


def aprova(d, s, h):
    m = (d.area_m2 <= AMAX) & (d.seg_prob_mean >= s)
    if h is not None:
        m &= ~(d.alt < h)
    return m


def main():
    if (OUT / "osm_teste_resultado.json").exists():
        raise SystemExit("teste do OSM com altura ja avaliado (avaliacao unica)")
    H = alturas()
    v, t = carregar("copas_val.csv", H), carregar("copas_teste.csv", H)
    print("sem altura (val/teste):", int(v.alt.isna().sum()), int(t.alt.isna().sum()))
    rows = []
    for h in HG:
        for s in SG:
            m = aprova(v, s, h)
            rows.append({"h_m": h, "s": s, "aprovadas": int(m.sum()),
                         "precisao": float(v[m].e_arvore.mean()) if m.any() else np.nan,
                         "arvores_mantidas": float(m[v.e_arvore].mean())})
    g = pd.DataFrame(rows); g.to_csv(OUT / "osm_validacao_grade.csv", index=False)
    ok = g[g.precisao >= META].copy()
    ok["h_ord"] = ok.h_m.fillna(-1)
    best = ok.sort_values(["aprovadas", "precisao", "h_ord", "s"], ascending=[False, False, True, True]).iloc[0]
    semh = ok[ok.h_m.isna()].sort_values(["aprovadas", "precisao", "s"], ascending=[False, False, True]).iloc[0]
    print("melhor sem altura (val):", semh[["s", "aprovadas", "precisao", "arvores_mantidas"]].to_dict())
    print("melhor geral (val):     ", best[["h_m", "s", "aprovadas", "precisao", "arvores_mantidas"]].to_dict())
    print("top 8 com altura:\n", ok[ok.h_m.notna()].sort_values("aprovadas", ascending=False).head(8)
          [["h_m", "s", "aprovadas", "precisao", "arvores_mantidas"]].round(3).to_string())
    adot = pd.notna(best.h_m)
    res = {"meta": META, "area_max_m2": AMAX, "val_sem_altura": semh.drop("h_ord").to_dict(),
           "val_escolhido": best.drop("h_ord").to_dict(), "altura_adotada": bool(adot)}

    def teste(s, h):
        m = aprova(t, s, h)
        nb = m & (t.pred_class != 5)
        return {"s": s, "h_m": h, "aprovadas": int(m.sum()), "precisao": float(t[m].e_arvore.mean()),
                "aprovadas_sem_bananeira": int(nb.sum()), "precisao_sem_bananeira": float(t[nb].e_arvore.mean()),
                "arvores_mantidas": float(m[t.e_arvore].mean())}
    res["teste_atual"] = teste(float(LIM["limiar_seg_prob_mean"]), None)
    if adot:
        res["teste_escolhido"] = teste(float(best.s), float(best.h_m))
    (OUT / "osm_teste_resultado.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in res if k.startswith("teste") or k == "altura_adotada"}, indent=2, default=float))


if __name__ == "__main__":
    main()
