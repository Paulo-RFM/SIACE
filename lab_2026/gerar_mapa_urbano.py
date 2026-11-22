#!/usr/bin/env python3
"""Mapa das arvores detectadas so na area urbana, delimitada por um perimetro em KML (2026-11-17).

Mesma ideia de gerar_mapa_cidade.py (que nao e alterado), restrita a um poligono definido pelo autor:
  * tiles: entram os tiles da grade de 512 px que INTERSECTAM o poligono (posicao de cada tile pela grade
    continua a partir do tile 001_001, como no pipeline); a lista vai para <saida_dir>/tiles_perimetro.csv;
  * ortofoto: remontada so com esses tiles, em --px-tile px por tile; fora do poligono fica esmaecida;
  * arvores: so as de arvores.csv cujo ponto (lon/lat) cai dentro do poligono; a legenda conta essas;
  * detalhe: janela de --det-tiles tiles (largura x altura) em resolucao maior, posicionada na regiao com
    mais arvores de ate 217,4 m2 (limite de area do OSM, para nao centrar o detalhe num macico de mata),
    ou em --det-centro lon,lat. O retangulo do detalhe e marcado no mapa geral.
Uso: python gerar_mapa_urbano.py --cidade-run runs/<cidade> --kml Placemark.kml [--saida <png>]
"""
import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MPath
from PIL import Image
from shapely.geometry import Polygon, box
from shapely.prepared import prep

from tree_georef import read_geotransform

LAB = Path(__file__).resolve().parent
TILES = Path("/home/paulomaciel/Datasets/drone_rp_v01/tiles512")
OUT = LAB / "LaTex" / "TCC2___Paulo_Maciel (1)" / "mapa_resultado_especies.png"
CLASSES = [(1, "Árvore sem identificação", "#2ca02c"), (2, "Palmeira", "#ff7f0e"), (3, "Árvore seca", "#8c564b"),
           (4, "Mangueira", "#9467bd"), (5, "Bananeira", "#ffd700"), (6, "Jabuticabeira", "#1f77b4")]
AREA_MAX_OSM = 217.3924945722511  # runs/osm_limiares_segformer_20261111/limiares_osm.json (so p/ posicionar o detalhe)
PIX_M = 0.0319


def ler_kml(path):
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    polys = []
    for c in ET.parse(path).getroot().iterfind(".//k:Polygon/k:outerBoundaryIs/k:LinearRing/k:coordinates", ns):
        pts = [tuple(float(v) for v in p.split(",")[:2]) for p in c.text.split()]
        polys.append(Polygon(pts))
    if len(polys) != 1:
        raise SystemExit(f"{path}: esperado 1 poligono, encontrados {len(polys)}")
    p = polys[0]
    return p if p.is_valid else p.buffer(0)


def ler_tile(r, c):
    f = TILES / f"OrtoRP-3cm_{r:03d}_{c:03d}.tif"
    if not f.exists():
        return None
    im = Image.open(f).convert("RGBA")
    if im.size != (512, 512):
        full = Image.new("RGBA", (512, 512), (0, 0, 0, 0)); full.paste(im, (0, 0)); im = full
    return im


def montar(tiles, r0, c0, nr, nc, k):
    mos = np.full((nr * k, nc * k, 3), 255, np.uint8)
    for r, c in tiles:
        im = ler_tile(r, c)
        if im is None:
            continue
        a = np.asarray(im.resize((k, k), Image.BOX)).astype(float)
        al = a[..., 3:4] / 255.0
        mos[(r - r0) * k:(r - r0 + 1) * k, (c - c0) * k:(c - c0 + 1) * k] = (a[..., :3] * al + 255 * (1 - al)).astype(np.uint8)
    return mos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cidade-run", required=True)
    ap.add_argument("--kml", required=True)
    ap.add_argument("--px-tile", type=int, default=20)
    ap.add_argument("--det-tiles", default="8x5", help="largura x altura da janela de detalhe, em tiles")
    ap.add_argument("--det-px-tile", type=int, default=128)
    ap.add_argument("--det-centro", default=None, help="lon,lat do centro do detalhe (padrao: regiao mais densa)")
    ap.add_argument("--saida", default=str(OUT))
    ap.add_argument("--titulo", default="Árvores na área urbana", help="titulo da legenda")
    ap.add_argument("--saida-dir", default=str(LAB / "runs" / "mapa_urbano_20261117"))
    args = ap.parse_args()
    run, k = Path(args.cidade_run), args.px_tile
    sd = Path(args.saida_dir); sd.mkdir(parents=True, exist_ok=True)

    poly = ler_kml(args.kml); pp = prep(poly)
    lon0, lat0, sx, sy = read_geotransform(TILES / "OrtoRP-3cm_001_001.tif")
    tw, th = 512 * sx, 512 * sy  # tamanho do tile em graus
    # tiles da grade (linha r, coluna c, a partir de 1) que intersectam o poligono
    minx, miny, maxx, maxy = poly.bounds
    c_a, c_b = int((minx - lon0) // tw) + 1, int((maxx - lon0) // tw) + 1
    r_a, r_b = int((lat0 - maxy) // th) + 1, int((lat0 - miny) // th) + 1
    sel = [(r, c) for r in range(r_a, r_b + 1) for c in range(c_a, c_b + 1)
           if pp.intersects(box(lon0 + (c - 1) * tw, lat0 - r * th, lon0 + c * tw, lat0 - (r - 1) * th))]
    exist = [(r, c) for r, c in sel if (TILES / f"OrtoRP-3cm_{r:03d}_{c:03d}.tif").exists()]
    pd.DataFrame(sel, columns=["linha", "coluna"]).assign(
        tile=[f"OrtoRP-3cm_{r:03d}_{c:03d}" for r, c in sel],
        existe=[(r, c) in set(exist) for r, c in sel]).to_csv(sd / "tiles_perimetro.csv", index=False)
    r0, c0 = min(r for r, _ in sel), min(c for _, c in sel)
    nr, nc = max(r for r, _ in sel) - r0 + 1, max(c for _, c in sel) - c0 + 1
    cache = sd / f"mosaico_{k}px_r{r0}-{r0 + nr - 1}_c{c0}-{c0 + nc - 1}.npy"
    if cache.exists():
        mos = np.load(cache)
    else:
        mos = montar(exist, r0, c0, nr, nc, k); np.save(cache, mos)

    def to_px(lon, lat, kk, rr0, cc0):
        return (np.asarray(lon) - lon0) / sx * kk / 512 - (cc0 - 1) * kk, (lat0 - np.asarray(lat)) / sy * kk / 512 - (rr0 - 1) * kk

    # esmaece fora do poligono
    px, py = to_px(*np.array(poly.exterior.coords).T, k, r0, c0)
    yy, xx = np.mgrid[0:mos.shape[0], 0:mos.shape[1]]
    dentro = MPath(np.c_[px, py]).contains_points(np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]).reshape(mos.shape[:2])
    vis = mos.copy(); vis[~dentro] = (0.25 * vis[~dentro] + 0.75 * 255).astype(np.uint8)

    t = pd.read_csv(run / "arvores.csv")
    from shapely.geometry import Point
    t = t[[pp.contains(Point(a, b)) for a, b in zip(t.lon, t.lat)]].reset_index(drop=True)
    x, y = to_px(t.lon, t.lat, k, r0, c0)

    # janela de detalhe
    dw, dh = (int(v) for v in args.det_tiles.lower().split("x"))
    if args.det_centro:
        lo, la = (float(v) for v in args.det_centro.split(","))
        cc, rc = int((lo - lon0) // tw) + 1, int((lat0 - la) // th) + 1
        dr0, dc0 = rc - dh // 2, cc - dw // 2
    else:
        s = t[t.area_m2 <= AREA_MAX_OSM]
        tr = ((lat0 - s.lat) // th).astype(int) + 1; tc = ((s.lon - lon0) // tw).astype(int) + 1
        H = np.zeros((nr, nc)); np.add.at(H, (tr - r0, tc - c0), 1)
        best, dr0, dc0 = -1, r0, c0
        for i in range(nr - dh + 1):
            for j in range(nc - dw + 1):
                v = H[i:i + dh, j:j + dw].sum()
                if v > best:
                    best, dr0, dc0 = v, r0 + i, c0 + j
    det_tiles = [(r, c) for r in range(dr0, dr0 + dh) for c in range(dc0, dc0 + dw)]
    kd = args.det_px_tile
    det = montar(det_tiles, dr0, dc0, dh, dw, kd)
    xd, yd = to_px(t.lon, t.lat, kd, dr0, dc0)
    nd = (xd >= 0) & (xd < dw * kd) & (yd >= 0) & (yd < dh * kd)

    # figura: mapa geral em cima, detalhe embaixo
    er_ = [r for r, _ in exist]; ec_ = [c for _, c in exist]
    H0, W0 = (max(er_) - min(er_) + 1) * k, (max(ec_) - min(ec_) + 1) * k  # proporcao da vista
    fw = 10
    fig = plt.figure(figsize=(fw, fw * H0 / W0 + fw * (dh * kd) / (dw * kd) + 0.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[H0 / W0, dh / dw], hspace=0.08)
    ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    ax.imshow(vis)
    ax.plot(px, py, color="#d62728", lw=1.0, zorder=5)
    handles = []
    for cid, nome, cor in CLASSES:
        sm = (t.pred_class == cid).to_numpy()
        ax.scatter(x[sm], y[sm], s=2.5 if cid == 1 else 5, c=cor, linewidths=0, zorder=3 if cid == 1 else 4)
        ax2.scatter(xd[sm & nd], yd[sm & nd], s=28, c=cor, edgecolors="k", linewidths=0.4, zorder=4)
        handles.append(Line2D([], [], marker="o", ls="", color=cor, label=f"{nome} ({sm.sum():,})".replace(",", ".")))
    ax.add_patch(Rectangle(((dc0 - c0) * k, (dr0 - r0) * k), dw * k, dh * k, fill=False, ec="k", lw=1.5, zorder=6))
    ax.text((dc0 - c0) * k, (dr0 - r0) * k - 4, "detalhe", fontsize=8, va="bottom", zorder=6)
    er = [r for r, _ in exist]; ec = [c for _, c in exist]
    X0, X1 = (min(ec) - c0) * k, (max(ec) - c0 + 1) * k
    Y0, Y1 = (min(er) - r0) * k, (max(er) - r0 + 1) * k
    ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.set_xticks([]); ax.set_yticks([])
    Wm = (X1 - X0) * 512 / k * PIX_M  # largura da vista em metros
    Lm = max(v for v in (10, 20, 50, 100, 200, 500, 1000) if v <= 0.25 * Wm)
    L = Lm / (512 / k * PIX_M)
    bx, by = X0 + 0.04 * (X1 - X0), Y0 + 0.96 * (Y1 - Y0)
    ax.plot([bx, bx + L], [by, by], "k-", lw=3)
    ax.text(bx + L / 2, by - 0.012 * (Y1 - Y0), f"{Lm} m", ha="center", va="bottom")
    xn = X0 + 0.96 * (X1 - X0)
    ax.annotate("N", xy=(xn, Y0 + 0.04 * (Y1 - Y0)), xytext=(xn, Y0 + 0.11 * (Y1 - Y0)), ha="center",
                arrowprops=dict(arrowstyle="-|>", color="k"), fontsize=12, fontweight="bold")
    ax.legend(handles=handles, loc="lower right", fontsize=8, markerscale=1.5, framealpha=0.9,
              title=f"{args.titulo} ({len(t):,})".replace(",", "."), title_fontsize=9)
    ax2.imshow(det); ax2.set_xlim(0, dw * kd); ax2.set_ylim(dh * kd, 0); ax2.set_xticks([]); ax2.set_yticks([])
    Ld = 20 / (512 / kd * PIX_M)
    ax2.plot([0.03 * dw * kd, 0.03 * dw * kd + Ld], [0.94 * dh * kd] * 2, "w-", lw=4)
    ax2.plot([0.03 * dw * kd, 0.03 * dw * kd + Ld], [0.94 * dh * kd] * 2, "k-", lw=2)
    ax2.text(0.03 * dw * kd + Ld / 2, 0.92 * dh * kd, "20 m", ha="center", va="bottom", fontsize=9,
             bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    fig.savefig(args.saida, dpi=200, bbox_inches="tight"); plt.close(fig)

    resumo = {"kml": str(args.kml), "cidade_run": str(run), "tiles_intersectam": len(sel), "tiles_com_imagem": len(exist),
              "arvores_no_perimetro": len(t), "arvores_no_run": int(len(pd.read_csv(run / "arvores.csv", usecols=["tree_id"]))),
              "por_classe": t.pred_name.value_counts().to_dict(),
              "detalhe_tiles": f"linhas {dr0}-{dr0 + dh - 1}, colunas {dc0}-{dc0 + dw - 1}", "arvores_no_detalhe": int(nd.sum())}
    import json
    (sd / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps(resumo, indent=2, ensure_ascii=False)); print("mapa salvo em", args.saida)


if __name__ == "__main__":
    main()
