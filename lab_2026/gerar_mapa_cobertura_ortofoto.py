#!/usr/bin/env python3
"""Mapa da cobertura de dossel (grade de 100 m) sobre a ortofoto do perimetro urbano (2026-11-20).

Para a apresentacao. Le as celulas de cobertura_dossel.py (celulas.geojson, WGS 84) e desenha cada celula
semitransparente sobre o mosaico de tiles do perimetro (Placemark.kml).
O mosaico parte do cache do mapa urbano (so tiles512) e completa os tiles ausentes com ~/proj/novas_imagens,
os mesmos usados no calculo da cobertura.
Uso: python gerar_mapa_cobertura_ortofoto.py [--celulas ...] [--saida ...]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Polygon as MPoly  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from PIL import Image  # noqa: E402
from pyproj import Geod  # noqa: E402

from gerar_mapa_urbano import ler_kml  # noqa: E402
from tree_georef import read_geotransform  # noqa: E402

LAB = Path(__file__).resolve().parent
OUT = LAB / "runs" / "cobertura_dossel_20261120"
URB = LAB / "runs" / "mapa_urbano_20261117"
DIRS = [Path("/home/paulomaciel/Datasets/drone_rp_v01/tiles512"), Path("/home/paulomaciel/proj/novas_imagens")]
K = 20  # px por tile no mosaico (mesmo do cache do mapa urbano)
CLASSES = [0, 5, 10, 20, 30, 101]
CORES = ["#c8553d", "#e9a97a", "#f3e3b3", "#8fc98a", "#2f7d4a"]
ROTULOS = ["< 5%", "5–10%", "10–20%", "20–30%", "≥ 30%"]


def ler_tile(stem):
    for d in DIRS:
        f = d / f"{stem}.tif"
        if f.exists():
            im = Image.open(f).convert("RGBA")
            if im.size != (512, 512):
                full = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
                full.paste(im, (0, 0))
                im = full
            return im
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celulas", default=str(OUT / "celulas.geojson"))
    ap.add_argument("--saida", default=str(OUT / "mapa_cobertura_ortofoto.png"))
    ap.add_argument("--alpha", type=float, default=0.55)
    a = ap.parse_args()

    tp = pd.read_csv(URB / "tiles_perimetro.csv")
    r0, c0 = tp.linha.min(), tp.coluna.min()
    nr, nc = tp.linha.max() - r0 + 1, tp.coluna.max() - c0 + 1
    cache = URB / f"mosaico_{K}px_r{r0}-{r0 + nr - 1}_c{c0}-{c0 + nc - 1}.npy"
    mos = np.load(cache)
    completados = 0
    for r in tp[~tp.existe].itertuples():  # tiles que faltavam em tiles512
        im = ler_tile(r.tile)
        if im is None:
            continue
        arr = np.asarray(im.resize((K, K), Image.BOX)).astype(float)
        al = arr[..., 3:4] / 255.0
        mos[(r.linha - r0) * K:(r.linha - r0 + 1) * K, (r.coluna - c0) * K:(r.coluna - c0 + 1) * K] = \
            (arr[..., :3] * al + 255 * (1 - al)).astype(np.uint8)
        completados += 1
    print(f"mosaico {mos.shape}, {completados} tiles completados com novas_imagens")

    lon0, lat0, sx, sy = read_geotransform(DIRS[0] / "OrtoRP-3cm_001_001.tif")

    def to_px(lon, lat):
        return ((np.asarray(lon) - lon0) / sx * K / 512 - (c0 - 1) * K,
                (lat0 - np.asarray(lat)) / sy * K / 512 - (r0 - 1) * K)

    poly = ler_kml(LAB / "Placemark.kml")
    px, py = to_px(*np.array(poly.exterior.coords).T)
    yy, xx = np.mgrid[0:mos.shape[0], 0:mos.shape[1]]
    dentro = MPath(np.c_[px, py]).contains_points(np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]).reshape(mos.shape[:2])
    vis = mos.copy()
    vis[~dentro] = (0.25 * vis[~dentro] + 0.75 * 255).astype(np.uint8)

    cmap, norm = ListedColormap(CORES), BoundaryNorm(CLASSES, len(CORES))
    h, w = vis.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 10 * h / w), dpi=220)
    ax.imshow(vis, interpolation="lanczos")
    feats = json.loads(Path(a.celulas).read_text())["features"]
    for f in feats:
        p = f["properties"]
        g = f["geometry"]
        aneis = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for an in aneis:
            x, y = to_px(*np.array(an[0]).T)
            if p["exibida"]:
                ax.add_patch(MPoly(np.c_[x, y], closed=True, fc=cmap(norm(p["cobertura_pct"])), alpha=a.alpha,
                                   ec="white", lw=0.35))
            else:
                ax.add_patch(MPoly(np.c_[x, y], closed=True, fc="none", ec="#777777", lw=0.3, hatch="////"))
    ax.plot(px, py, color="#222222", lw=1.1)
    ax.set_xlim(px.min() - 15, px.max() + 15)
    ax.set_ylim(py.max() + 60, py.min() - 15)
    ax.set_axis_off()
    # escala: metros por pixel do mosaico, medido na linha central do perimetro
    lat_c = poly.centroid.y
    m_por_px = Geod(ellps="WGS84").inv(lon0, lat_c, lon0 + 512 * sx, lat_c)[2] / K
    L = 500
    xb, yb = px.min(), py.max() + 25
    ax.plot([xb, xb + L / m_por_px], [yb, yb], color="black", lw=3)
    ax.text(xb + L / m_por_px / 2, yb - 30, f"{L} m", ha="center", va="bottom", fontsize=11)
    ax.annotate("N", xy=(px.max() - 60, py.min() + 10), xytext=(px.max() - 60, py.min() + 330), ha="center", fontsize=13,
                fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    hs = [Patch(fc=c, alpha=a.alpha, ec="#555555", label=r) for c, r in zip(CORES, ROTULOS)] + [
        Patch(fc="white", hatch="////", ec="#777777", label="Borda (não avaliada)")]
    ax.legend(handles=hs, title="Cobertura de copa", loc="upper right", bbox_to_anchor=(1.0, 0.82),
              fontsize=10, title_fontsize=11, frameon=True)
    fig.savefig(a.saida, bbox_inches="tight", facecolor="white")
    print("mapa:", a.saida)


if __name__ == "__main__":
    main()
