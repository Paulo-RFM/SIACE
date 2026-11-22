#!/usr/bin/env python3
"""Filtro de altura para falsas arvores (2026-11-18). Criterio em runs/filtro_altura_20261118/criterio.md.

Etapa "val": escolhe o limiar h na validacao (maior F1 de deteccao) e grava limiar_altura.json.
Etapa "teste": aplica o h congelado no teste UMA vez (recusa se ja existir resultado) e calcula IC 95% por
bootstrap por tile do F1 com e sem filtro e da diferenca pareada.
A deteccao usa o mesmo casamento do evaluate_pipeline_test.py (gt_polygons, hungaro, IoU >= 0,5).
Uso: python filtro_altura.py val | teste
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

from evaluate_pipeline_test import gt_polygons

LAB = Path(__file__).resolve().parent
R = LAB / "runs"
OUT = R / "filtro_altura_20261118"
GEO = Path("/home/paulomaciel/proj/arvores com altura.geojson")
CITY = R / "pipeline_cidade_segformer_20261111"
RUNS = {"val": R / "pipeline_val_hmax_segformer_20261111", "teste": R / "pipeline_test_hmax_segformer_20261111"}
GRADE = [None] + [round(0.25 * k, 2) for k in range(1, 13)]
B, SEED = 2000, 42


def alturas():
    g = json.loads(GEO.read_text())
    d = pd.DataFrame([f["properties"] for f in g["features"]])
    cid = pd.read_csv(CITY / "arvores.csv", usecols=["tree_id", "parts"])
    if not (d.tree_id.to_numpy() == cid.tree_id.to_numpy()).all() or not (d.parts == cid.parts).all():
        raise SystemExit("geojson de alturas nao corresponde a arvores.csv da cidade")
    h = {}
    for parts, a in zip(d.parts, d.alt_1):
        for p in parts.split(";"):
            t, i = p.split("#")
            h[(t, int(i))] = a
    return h


def preparar(run, split, H):
    sp = pd.read_csv(Path("/home/paulomaciel/proj/unet_spatial_split/tiles_split.csv"))
    tiles = sorted(sp[sp.split == split].tile)
    dados = []
    for t in tiles:
        ip = run / "mascaras" / "instances" / f"{t}_instances.png"
        inst = np.array(Image.open(ip)).astype(np.int32) if ip.is_file() else np.zeros((512, 512), np.int32)
        ids = [int(i) for i in np.unique(inst) if i > 0]
        gts = [g for g, _ in gt_polygons(t)]
        M = np.zeros((len(gts), len(ids)))
        for b, i in enumerate(ids):
            p = inst == i
            for a, g in enumerate(gts):
                inter = (g & p).sum()
                if inter:
                    M[a, b] = inter / (g | p).sum()
        alt = np.array([H.get((t, i), np.nan) for i in ids], float)
        dados.append((t, M, alt))
    return dados


def det_tile(M, keep):
    n_gt, n_pred = M.shape[0], int(keep.sum())
    tp = 0
    if n_gt and n_pred:
        Mk = M[:, keep]
        r, c = linear_sum_assignment(-Mk)
        tp = int((Mk[r, c] >= 0.5).sum())
    return n_gt, n_pred, tp


def tabela(dados, h):
    rows = []
    for t, M, alt in dados:
        keep = np.ones(len(alt), bool) if h is None else ~(alt < h)  # NaN -> mantida
        rows.append((t, *det_tile(M, keep)))
    return pd.DataFrame(rows, columns=["tile", "n_gt", "n_pred", "tp"])


def f1(d):
    return 2 * d.tp.sum() / max(d.n_pred.sum() + d.n_gt.sum(), 1)


def resumo(d):
    P, G, T = d.n_pred.sum(), d.n_gt.sum(), d.tp.sum()
    return {"instancias": int(P), "tp": int(T), "precisao": T / max(P, 1), "recall": T / max(G, 1), "f1": f1(d),
            "instancias_por_arvore": P / max(G, 1)}


def main():
    etapa = sys.argv[1]
    OUT.mkdir(exist_ok=True)
    H = alturas()
    if etapa == "val":
        dados = preparar(RUNS["val"], "val", H)
        rows = [{"h_m": h, **resumo(tabela(dados, h))} for h in GRADE]
        cur = pd.DataFrame(rows)
        cur.to_csv(OUT / "validacao_curva.csv", index=False)
        print(cur.round(4).to_string())
        base = cur.f1.iloc[0]
        best = cur.iloc[1:].sort_values(["f1", "h_m"], ascending=[False, True]).iloc[0]
        adot = bool(best.f1 > base)
        res = {"h_m": float(best.h_m) if adot else None, "f1_val": float(best.f1), "f1_val_sem_filtro": float(base),
               "adotado": adot, "criterio": "criterio.md", "nan": "mantidas"}
        (OUT / "limiar_altura.json").write_text(json.dumps(res, indent=2))
        print(res)
    elif etapa == "teste":
        if (OUT / "teste_resultado.json").exists():
            raise SystemExit("teste ja avaliado (avaliacao unica)")
        lim = json.loads((OUT / "limiar_altura.json").read_text())
        if not lim["adotado"]:
            raise SystemExit("filtro nao adotado na validacao; teste nao avaliado")
        dados = preparar(RUNS["teste"], "test", H)
        a, b = tabela(dados, None), tabela(dados, lim["h_m"])
        rng = np.random.default_rng(SEED)
        idx = [rng.integers(0, len(a), len(a)) for _ in range(B)]
        fa = np.array([f1(a.iloc[i]) for i in idx]); fb = np.array([f1(b.iloc[i]) for i in idx])
        ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
        res = {"h_m": lim["h_m"], "sem_filtro": {**resumo(a), "f1_ic95": ci(fa)},
               "com_filtro": {**resumo(b), "f1_ic95": ci(fb)},
               "dif_f1": f1(b) - f1(a), "dif_f1_ic95": ci(fb - fa), "n_tiles": len(a)}
        (OUT / "teste_resultado.json").write_text(json.dumps(res, indent=2, default=float))
        b.to_csv(OUT / "teste_deteccao_por_tile_com_filtro.csv", index=False)
        print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
