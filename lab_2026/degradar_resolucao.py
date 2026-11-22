#!/usr/bin/env python3
"""Simula imagens de resolução menor a partir dos tiles anotados de 3,19 cm (experimento de resolução, 2026-11-21).

Para cada tile de --entrada (512 x 512, RGBA, GeoTIFF EPSG:4326):
  1. reduz as bandas RGB para n x n px por média de área (Image.BOX), como um sensor de pixel maior;
     n = round(512 * 0,0319 / GSD alvo): 204 px para 8 cm (pixel efetivo 0,0801 m) e 109 px para 15 cm (0,1498 m);
  2. amplia de volta para 512 x 512 por interpolação bilinear, na mesma grade de 3,19 cm.
A grade, o georreferenciamento (copiado com GDAL), a banda alfa, as máscaras e a divisão espacial ficam iguais;
só o detalhe da imagem muda. Os pixels sem dado do original (RGB = 0) continuam RGB = 0, e um pixel válido que
fique exatamente RGB = 0 recebe 1, para que valid_pixel_mask marque os mesmos pixels que no original.
Uso: python degradar_resolucao.py --entrada /home/paulomaciel/proj/imagens --saida <pasta> --gsd 0.08
"""
import argparse
import json
from pathlib import Path

import numpy as np
from osgeo import gdal
from PIL import Image

gdal.UseExceptions()
PIXEL_M = 0.0319  # GSD medido do ortomosaico (mesmo valor de separate_tree_crowns_whatershed.PIXEL_M)
T = 512


def degradar(rgb, n):
    out = np.empty_like(rgb)
    for b in range(rgb.shape[0]):
        im = Image.fromarray(rgb[b]).resize((n, n), Image.BOX).resize((T, T), Image.BILINEAR)
        out[b] = np.asarray(im)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--gsd", type=float, required=True, help="tamanho do pixel simulado (m)")
    a = ap.parse_args()
    n = int(round(T * PIXEL_M / a.gsd))
    out = Path(a.saida)
    out.mkdir(parents=True, exist_ok=False)  # não sobrescreve um conjunto existente
    tiles = sorted(Path(a.entrada).glob("*.tif"))
    drv = gdal.GetDriverByName("GTiff")
    for i, f in enumerate(tiles):
        src = gdal.Open(str(f))
        assert (src.RasterXSize, src.RasterYSize) == (T, T), f
        arr = src.ReadAsArray()
        rgb = arr[:3]
        nodata = (rgb == 0).all(axis=0)
        deg = degradar(rgb, n)
        deg[:, nodata] = 0
        zerou = (deg == 0).all(axis=0) & ~nodata
        deg[0, zerou] = 1
        dst = drv.CreateCopy(str(out / f.name), src)
        for b in range(3):
            dst.GetRasterBand(b + 1).WriteArray(deg[b])
        dst.FlushCache()
        dst = src = None
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(tiles)}", flush=True)
    info = {"entrada": a.entrada, "gsd_alvo_m": a.gsd, "pixel_origem_m": PIXEL_M, "n_reduzido_px": n,
            "gsd_efetivo_m": T * PIXEL_M / n, "metodo": "BOX (média de área) para n x n, BILINEAR de volta a 512",
            "n_tiles": len(tiles)}
    (out / "degradacao.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(info)


if __name__ == "__main__":
    main()
