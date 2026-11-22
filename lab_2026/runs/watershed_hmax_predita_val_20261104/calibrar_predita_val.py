"""Calibracao da individualizacao sobre mascaras PREDITAS pela U-Net principal (limiar 0,65), somente VALIDACAO.

Responde a P8 e P1(c) da revisao critica:
  * watershed original (marcadores = maximos locais da distancia em janela peak_footprint): grade
    completa peak_footprint x min_area (antes so min_area=1000 tinha sido varrido nas mascaras preditas);
  * watershed com marcadores h-maxima da transformada de distancia: um marcador por "pico" que se
    eleva pelo menos h acima do vale que o separa do vizinho. h e dado em metros (pixel ~ 3,19 cm),
    o que da significado fisico: duas copas so sao separadas se o "pescoco" entre elas for pelo menos
    h mais estreito que o raio de cada uma;
  * componentes conexas (referencia ja congelada, min_area 3000), para comparacao na mesma execucao.
Mesma limpeza morfologica, mesmo casamento (Hungaro, IoU >= 0,5) e mesmos 203 tiles de validacao.
Rodar a partir de lab_2026/. Saida: resultado.csv / resultado.json nesta pasta. O teste NAO e tocado.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import morphology, segmentation

sys.path.insert(0, str(Path.cwd()))
import separate_tree_crowns_whatershed as ws  # noqa: E402

P = Path("/home/paulomaciel/proj")
HERE = Path(__file__).resolve().parent
MASKS = P / "lab_2026/runs/watershed_vs_cc_val_20261103/pipeline_val_predita/mascaras/binary"
PIXEL_M = 0.0319


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


CONFIGS = ([("cc", None, 3000)]
           + [("watershed", fp, a) for fp in (41, 75, 121, 201, 251) for a in (1000, 2000, 3000)]
           + [("hmax", h, a) for h in (0.10, 0.20, 0.30, 0.50, 0.75, 1.00) for a in (1000, 2000, 3000)])


def main():
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
                         h_m=par if meth == "hmax" else None,
                         h_px=round(par / PIXEL_M) if meth == "hmax" else None, min_area=a,
                         instancias=n_tot, tp=tp, precisao=pr, recall=rc,
                         f1=2 * pr * rc / max(pr + rc, 1e-9), instancias_por_arvore=n_tot / G, mae_contagem=mae))
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "resultado.csv", index=False)
    best = {m: d[d.metodo == m].sort_values("f1").iloc[-1].to_dict() for m in d.metodo.unique()}
    (HERE / "resultado.json").write_text(json.dumps({"n_val_tiles": len(tiles), "n_gt": G, "melhor_por_metodo": best},
                                                    indent=1, default=str), encoding="utf-8")
    print(d.round(3).to_string())


if __name__ == "__main__":
    main()
