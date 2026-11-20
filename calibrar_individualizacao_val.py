#!/usr/bin/env python3
"""Calibracao da individualizacao das copas sobre mascaras PREDITAS, somente na VALIDACAO (203 tiles).

Reune numa so execucao o que antes estava em runs/watershed_vs_cc_val_20261103/comparar_predita_val*.py e
runs/watershed_hmax_predita_val_20261104/calibrar_predita_val.py (mesma limpeza morfologica, mesmo casamento
Hungaro com IoU >= 0,5, mesmos polígonos de labels/ e labels2/), para as mascaras de um novo modelo:
  * componentes conexas, min_area 500-6000 px;
  * watershed com marcadores = maximos locais da distancia (janela peak_footprint 41-251 px);
  * watershed com marcadores h-maxima (h 0,10-1,00 m).
Criterio fixado antes (o mesmo de 2026-11-03): componentes conexas com o min_area de maior F1. O arquivo
frozen_instance_params.json registra a escolha e se algum watershed superou as componentes conexas.
O teste nao e tocado. Rodar a partir de lab_2026/.
"""
import argparse
import json
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import morphology, segmentation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import separate_tree_crowns_whatershed as ws  # noqa: E402

P = Path("/home/paulomaciel/proj")
PIXEL_M = 0.0319
CONFIGS = ([("cc", None, a) for a in (500, 1000, 2000, 3000, 4000, 6000)]
           + [("watershed", fp, a) for fp in (41, 75, 121, 201, 251) for a in (1000, 2000, 3000)]
           + [("hmax", h, a) for h in (0.10, 0.20, 0.30, 0.50, 0.75, 1.00) for a in (1000, 2000, 3000)])
MASKS = None


def hmax_watershed(mask, h_px, min_area):
    clean = ws.clean_binary_mask(mask)
    if clean.sum() == 0:
        return np.zeros(mask.shape, np.int32)
    dist = ndi.distance_transform_edt(clean)
    markers = ndi.label(morphology.h_maxima(dist, h_px))[0]
    lab = segmentation.watershed(-dist, markers, mask=clean)
    ids, sizes = np.unique(lab[lab > 0], return_counts=True)
    keep = np.zeros(lab.max() + 1, np.int32)
    big = ids[sizes >= min_area]
    keep[big] = np.arange(1, len(big) + 1)
    return keep[lab]


def gts_of(t):
    jp = next((q for q in [P / "labels" / f"{t}.json", P / "labels2" / f"{t}.json"] if q.exists()), None)
    out = []
    if jp:
        for sh in json.load(open(jp))["shapes"]:
            if sh.get("label") != "tree" or len(sh["points"]) < 3:
                continue
            m = np.zeros((512, 512), np.uint8)
            cv2.fillPoly(m, [np.array(sh["points"], np.int32)], 1)
            out.append(m.astype(bool))
    return out


def score(inst, gts):
    ids = [i for i in np.unique(inst) if i > 0]
    if not gts or not ids:
        return len(ids), 0
    M = np.zeros((len(gts), len(ids)))
    for b, i in enumerate(ids):
        p = inst == i
        for a, g in enumerate(gts):
            it = (g & p).sum()
            if it:
                M[a, b] = it / (g | p).sum()
    r, c = linear_sum_assignment(-M)
    return len(ids), int((M[r, c] >= 0.5).sum())


def run_tile(t):
    gts = gts_of(t)
    m = (np.array(Image.open(MASKS / f"{t}.png")) > 127).astype(np.uint8)
    res = {}
    for cfg in CONFIGS:
        meth, par, a = cfg
        if meth == "cc":
            inst = ws.label_connected_components(m, min_area=a)[0]
        elif meth == "watershed":
            inst = ws.segment_tree_crowns(m, min_area=a, peak_footprint=par)[0]
        else:
            inst = hmax_watershed(m, round(par / PIXEL_M), a)
        res[cfg] = score(inst, gts) + (len(gts),)
    return res


def main():
    global MASKS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masks", required=True, help="mascaras binarias preditas da validacao (pipeline --save-masks)")
    ap.add_argument("--seg-run", required=True, help="run de segmentacao que gerou as mascaras (registro)")
    ap.add_argument("--saida", required=True)
    args = ap.parse_args()
    MASKS = Path(args.masks)
    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(P / "unet_spatial_split/tiles_split.csv")
    tiles = list(s[s.split == "val"].tile)
    with Pool(3) as pool:
        per_tile = pool.map(run_tile, tiles)
    G = sum(next(iter(r.values()))[2] for r in per_tile)
    rows = []
    for cfg in CONFIGS:
        meth, par, a = cfg
        n_tot = sum(r[cfg][0] for r in per_tile)
        tp = sum(r[cfg][1] for r in per_tile)
        mae = float(np.mean([abs(r[cfg][0] - r[cfg][2]) for r in per_tile]))
        pr, rc = tp / max(n_tot, 1), tp / G
        rows.append(dict(metodo=meth, peak_footprint=par if meth == "watershed" else None,
                         h_m=par if meth == "hmax" else None, min_area=a, instancias=n_tot, tp=tp,
                         precisao=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-9),
                         instancias_por_arvore=n_tot / G, mae_contagem=mae))
    d = pd.DataFrame(rows)
    d.to_csv(out / "resultado.csv", index=False)
    best = {m: d[d.metodo == m].sort_values("f1").iloc[-1].to_dict() for m in d.metodo.unique()}
    cc_best = best["cc"]
    ws_beats = [m for m in ("watershed", "hmax") if best[m]["f1"] > cc_best["f1"]]
    frozen = {"method": "componentes_conexas", "min_area": int(cc_best["min_area"]), "connectivity": 4,
              "cleaning": {"closing_disk_radius": 5, "remove_small_holes_area": 32, "remove_small_objects_min_size": 32},
              "criterion": "max F1 deteccao (IoU>=0.5) na validacao, mascaras PREDITAS pelo modelo de segmentacao",
              "segmentation_run": args.seg_run, "f1_val": cc_best["f1"],
              "watershed_superou_cc": ws_beats, "selected_on": f"val ({len(tiles)} tiles, {G} arvores)",
              "frozen_at": str(date.today())}
    (out / "frozen_instance_params.json").write_text(json.dumps(frozen, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "resultado.json").write_text(json.dumps({"n_val_tiles": len(tiles), "n_gt": G, "seg_run": args.seg_run,
                                                    "melhor_por_metodo": best}, indent=1, default=str), encoding="utf-8")
    print(d.round(3).to_string())
    print(json.dumps(frozen, indent=1, ensure_ascii=False))
    sys.exit(3 if ws_beats else 0)


if __name__ == "__main__":
    main()
