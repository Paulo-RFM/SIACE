#!/usr/bin/env python3
"""Mapas de João Pessoa com as árvores detectadas pelo SIACE (experimento de resolução, 2026-11-22).

1. mapa_cidade.png: a ortoimagem inteira remontada em baixa resolução (a partir do arquivo original, EPSG:31985),
   com um ponto por árvore do modelo do sistema (SegFormer 3 cm) e o retângulo do recorte detalhado.
2. mapa_detalhe.png: recorte de 150 m x 150 m na resolução original (8 cm), na área com mais árvores
   isoladas (copa < 100 m² no modelo do sistema; evita fragmentos de mata, onde as copas se fundem),
   com as copas e os pontos de cada um dos 3 modelos lado a lado.
Os pontos e as copas vêm de resultados/joao_pessoa/<modelo>/ (WGS 84) e são reprojetados para EPSG:31985.
3. Versões georreferenciadas (abrem no QGIS sobre o .tif): mapa_cidade_geo.png e mapa_detalhe_<modelo>_geo.png,
   só a imagem com pontos e copas desenhados nos pixels, sem título nem margem, com .pgw (world file) e
   .png.aux.xml (EPSG:31985). As figuras 1 e 2 têm título e margens e não são georreferenciáveis.
A classe predita não é mostrada: as 6 classes são de Rio Paranaíba e não valem em João Pessoa.
Uso: python mapa_joao_pessoa.py   (ambiente conda "geo")
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from osgeo import gdal  # noqa: E402
from pyproj import Transformer  # noqa: E402

gdal.UseExceptions()
BASE = Path(__file__).resolve().parent
SRC = BASE / "6a90442d7efa87881ba82682.tif"
RES = BASE / "resultados" / "joao_pessoa"
MODELOS = {"atual": ("SegFormer 3 cm (sistema)", "#00e676"), "8cm": ("SegFormer 8 cm", "#ffb300"),
           "15cm": ("SegFormer 15 cm", "#ff1744")}
TO_UTM = Transformer.from_crs(4326, 31985, always_xy=True).transform
LADO = 150.0  # m, lado do recorte detalhado


def ler_raster(**kw):
    ds = gdal.Translate("", str(SRC), format="MEM", **kw)
    return np.moveaxis(ds.ReadAsArray(), 0, -1), ds.GetGeoTransform()


def pontos(m, area_max=None):
    fs = json.loads((RES / m / "arvores.geojson").read_text())["features"]
    if area_max is not None:
        fs = [f for f in fs if f["properties"]["area_m2"] < area_max]
    xy = np.array([f["geometry"]["coordinates"] for f in fs])
    return np.column_stack(TO_UTM(xy[:, 0], xy[:, 1]))


def copas(m):
    out = []
    for f in json.loads((RES / m / "copas.geojson").read_text())["features"]:
        g = f["geometry"]
        for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            a = np.array(poly[0])
            out.append(np.column_stack(TO_UTM(a[:, 0], a[:, 1])))
    return out


def salvar_geo(nome, img, gt, pts_utm, copas_utm, cor, raio):
    """Desenha copas (contorno) e pontos nos pixels da imagem e grava PNG + .pgw + .png.aux.xml (EPSG:31985)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.ascontiguousarray(img[..., :3]))
    d = ImageDraw.Draw(im)
    for c in copas_utm or []:
        x, y = para_px(c, gt)
        if x.max() < 0 or x.min() > im.width or y.max() < 0 or y.min() > im.height:
            continue
        d.line(list(zip(x, y)), fill=cor, width=max(1, raio // 3))
    x, y = para_px(pts_utm, gt)
    for xi, yi in zip(x, y):
        if 0 <= xi < im.width and 0 <= yi < im.height:
            d.ellipse([xi - raio, yi - raio, xi + raio, yi + raio], fill=cor, outline="black")
    arr = np.moveaxis(np.asarray(im), -1, 0)
    mem = gdal.GetDriverByName("MEM").Create("", im.width, im.height, 3, gdal.GDT_Byte)
    mem.SetGeoTransform(gt)
    mem.SetProjection(gdal.Open(str(SRC)).GetProjection())
    for b in range(3):
        mem.GetRasterBand(b + 1).WriteArray(arr[b])
    out = RES / f"{nome}.png"
    gdal.GetDriverByName("PNG").CreateCopy(str(out), mem)  # grava também o .png.aux.xml com o SRC
    # world file: tamanho do pixel, rotações e centro do pixel superior esquerdo
    out.with_suffix(".pgw").write_text("\n".join(f"{v:.10f}" for v in (
        gt[1], gt[4], gt[2], gt[5], gt[0] + gt[1] / 2 + gt[2] / 2, gt[3] + gt[4] / 2 + gt[5] / 2)) + "\n")
    return out


def para_px(xy, gt):
    return (xy[:, 0] - gt[0]) / gt[1], (xy[:, 1] - gt[3]) / gt[5]


def main():
    pts = {m: pontos(m) for m in MODELOS}
    # área do recorte: célula de 150 m com mais árvores isoladas (copa < 100 m²) do modelo do sistema
    p = pts["atual"]
    iso = pontos("atual", area_max=100)
    x0, y0 = p[:, 0].min(), p[:, 1].min()
    ij = np.floor((iso - [x0, y0]) / LADO).astype(int)
    cel, n = np.unique(ij, axis=0, return_counts=True)
    ci, cj = cel[n.argmax()]
    cx, cy = x0 + (ci + 0.5) * LADO, y0 + (cj + 0.5) * LADO
    win = [cx - LADO / 2, cy + LADO / 2, cx + LADO / 2, cy - LADO / 2]  # ulx, uly, lrx, lry

    img, gt = ler_raster(width=3000)
    fig, ax = plt.subplots(figsize=(12, 12 * img.shape[0] / img.shape[1]))
    ax.imshow(img)
    px, py = para_px(p, gt)
    ax.scatter(px, py, s=7, c=MODELOS["atual"][1], edgecolors="black", linewidths=0.25, alpha=0.95)
    rx, ry = para_px(np.array([[win[0], win[1]]]), gt)
    ax.add_patch(Rectangle((rx[0], ry[0]), LADO / gt[1], LADO / -gt[5], fill=False, ec="red", lw=1.8))
    m500 = 500 / gt[1]
    ax.plot([60, 60 + m500], [img.shape[0] - 60] * 2, color="black", lw=4)
    ax.text(60 + m500 / 2, img.shape[0] - 80, "500 m", ha="center", fontsize=12)
    ax.set_axis_off()
    n_fmt = f"{len(p):,}".replace(",", ".")
    ax.set_title(f"João Pessoa: {n_fmt} árvores detectadas pelo SIACE (SegFormer 3 cm), um ponto por árvore",
                 fontsize=14)
    fig.savefig(RES / "mapa_cidade.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    det, gtd = ler_raster(projWin=win)
    fig, axs = plt.subplots(1, 3, figsize=(21, 7.4))
    for ax, (m, (nome, cor)) in zip(axs, MODELOS.items()):
        ax.imshow(det)
        for c in copas(m):
            x, y = para_px(c, gtd)
            if x.max() < 0 or x.min() > det.shape[1] or y.max() < 0 or y.min() > det.shape[0]:
                continue
            ax.plot(x, y, color=cor, lw=1.2)
        x, y = para_px(pts[m], gtd)
        dentro = (x >= 0) & (x < det.shape[1]) & (y >= 0) & (y < det.shape[0])
        ax.scatter(x[dentro], y[dentro], s=28, c=cor, edgecolors="black", linewidths=0.6, zorder=3)
        ax.set_xlim(0, det.shape[1])
        ax.set_ylim(det.shape[0], 0)
        ax.set_title(f"{nome}: {dentro.sum()} árvores", fontsize=14)
        ax.set_axis_off()
    m20 = 20 / gtd[1]
    axs[0].plot([30, 30 + m20], [det.shape[0] - 40] * 2, color="white", lw=4)
    axs[0].text(30 + m20 / 2, det.shape[0] - 60, "20 m", ha="center", color="white", fontsize=12)
    fig.suptitle(f"Recorte de {LADO:.0f} m × {LADO:.0f} m (imagem original de 8 cm): copas e pontos de cada modelo",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(RES / "mapa_detalhe.png", dpi=150, facecolor="white")
    plt.close(fig)
    salvar_geo("mapa_cidade_geo", img, gt, p, None, MODELOS["atual"][1], raio=3)
    for m, (nome, cor) in MODELOS.items():
        salvar_geo(f"mapa_detalhe_{m}_geo", det, gtd, pts[m], copas(m), cor, raio=10)
    print("mapas:", RES / "mapa_cidade.png", RES / "mapa_detalhe.png", "+ versões _geo.png com .pgw")


if __name__ == "__main__":
    main()
