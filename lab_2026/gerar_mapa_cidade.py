#!/usr/bin/env python3
"""Gera mapa_resultado_especies.png (monografia, fig:mapa_resultado): arvores detectadas sobre a ortofoto.

A ortofoto e remontada em baixa resolucao a partir dos proprios tiles GeoTIFF (cada tile 512 px reduzido a
--px-tile px; mosaico em cache .npy em runs/mosaico_ortofoto_cache/ para nao reler os 61 mil tiles). Os pontos sao as arvores
de arvores.csv (lon/lat em WGS 84), coloridas pela classe predita. Posicionamento pelo georreferenciamento do
tile 001_001 (tree_georef.read_geotransform) e pela grade continua de 512 px entre tiles.
Uso: python gerar_mapa_cidade.py --cidade-run runs/<cidade> [--saida <png>]
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

from tree_georef import read_geotransform

LAB = Path(__file__).resolve().parent
TILES = Path("/home/paulomaciel/Datasets/drone_rp_v01/tiles512")
OUT = LAB / "LaTex" / "TCC2___Paulo_Maciel (1)" / "mapa_resultado_especies.png"
CLASSES = [(1, "Árvore sem identificação", "#2ca02c"), (2, "Palmeira", "#ff7f0e"), (3, "Árvore seca", "#8c564b"),
           (4, "Mangueira", "#9467bd"), (5, "Bananeira", "#ffd700"), (6, "Jabuticabeira", "#1f77b4")]


def mosaico(cache, k):
    if cache.exists():
        return np.load(cache)
    files = sorted(TILES.glob("OrtoRP-3cm_*_*.tif"))
    rc = [tuple(int(x) for x in re.findall(r"_(\d+)_(\d+)\.tif$", f.name)[0]) for f in files]
    R, C = max(r for r, _ in rc), max(c for _, c in rc)
    mos = np.full((R * k, C * k, 3), 255, np.uint8)
    for f, (r, c) in zip(files, rc):
        im = Image.open(f).convert("RGBA")
        if im.size != (512, 512):  # ultima linha/coluna: completa com sem-dado, como no pipeline
            full = Image.new("RGBA", (512, 512), (0, 0, 0, 0)); full.paste(im, (0, 0)); im = full
        a = np.asarray(im.resize((k, k), Image.BOX)).astype(float)
        alpha = a[..., 3:4] / 255.0
        mos[(r - 1) * k:r * k, (c - 1) * k:c * k] = (a[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    np.save(cache, mos)
    return mos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cidade-run", required=True)
    ap.add_argument("--px-tile", type=int, default=8)
    ap.add_argument("--saida", default=str(OUT))
    args = ap.parse_args()
    run, k = Path(args.cidade_run), args.px_tile
    cache = LAB / "runs" / "mosaico_ortofoto_cache"; cache.mkdir(exist_ok=True)
    mos = mosaico(cache / f"mosaico_{k}px.npy", k)
    lon0, lat0, sx, sy = read_geotransform(TILES / "OrtoRP-3cm_001_001.tif")
    t = pd.read_csv(run / "arvores.csv")
    x = (t.lon - lon0) / sx * k / 512
    y = (lat0 - t.lat) / sy * k / 512
    occ = np.where((mos < 250).any(axis=2))
    y0, y1, x0, x1 = occ[0].min(), occ[0].max(), occ[1].min(), occ[1].max()
    fig, ax = plt.subplots(figsize=(10, 10 * (y1 - y0) / max(x1 - x0, 1) + 0.6))
    ax.imshow(mos, alpha=0.75)
    handles = []
    for cid, nome, cor in CLASSES:
        s = t.pred_class == cid
        ax.scatter(x[s], y[s], s=1.2 if cid == 1 else 3, c=cor, linewidths=0, zorder=3 if cid == 1 else 4)
        handles.append(Line2D([], [], marker="o", ls="", color=cor, label=f"{nome} ({s.sum():,})".replace(",", ".")))
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_xticks([]); ax.set_yticks([])
    # barra de escala: 500 m (pixel de 3,19 cm no tile)
    m_por_px_mapa = 512 / k * 0.0319
    L = 500 / m_por_px_mapa
    bx, by = x0 + 0.04 * (x1 - x0), y1 - 0.04 * (y1 - y0)
    ax.plot([bx, bx + L], [by, by], "k-", lw=3); ax.text(bx + L / 2, by - 0.012 * (y1 - y0), "500 m", ha="center", va="bottom")
    ax.annotate("N", xy=(x1 - 0.04 * (x1 - x0), y0 + 0.04 * (y1 - y0)), xytext=(x1 - 0.04 * (x1 - x0), y0 + 0.11 * (y1 - y0)),
                ha="center", arrowprops=dict(arrowstyle="-|>", color="k"), fontsize=12, fontweight="bold")
    ax.legend(handles=handles, loc="lower right", fontsize=8, markerscale=1.5, framealpha=0.9,
              title=f"Árvores detectadas ({len(t):,})".replace(",", "."), title_fontsize=9)
    fig.tight_layout(); fig.savefig(args.saida, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("mapa salvo em", args.saida, "| arvores:", len(t))


if __name__ == "__main__":
    main()
