#!/usr/bin/env python3
"""Recorta uma ortoimagem GeoTIFF em tiles de 512 x 512 px, sem sobreposição, com GDAL.

- A grade começa no canto superior esquerdo da imagem; o tile (linha, coluna) cobre os pixels
  [(linha-1)*T, linha*T) x [(coluna-1)*T, coluna*T), com T = --tamanho (padrão 512).
- Os tiles da última linha e da última coluna são completados com branco (255 em todas as bandas)
  até T x T. A imagem original não é reamostrada nem alterada.
- Cada tile recebe o sistema de coordenadas da imagem de entrada e a sua própria geotransformação
  (origem deslocada pelo deslocamento em pixels), de modo que o georreferenciamento é preservado.
  O pixel do tile ocupa exatamente a mesma posição no terreno que o pixel correspondente na imagem.
- Nome dos tiles: <prefixo>_<linha>_<coluna>.tif, linha e coluna a partir de 1, com zeros à esquerda
  (mesmo padrão dos tiles de Rio Paranaíba, ex.: OrtoRP-3cm_001_001.tif).
- Gera também <saida>/indice_tiles.csv com a posição de cada tile e a fração de pixels completados.
- --pular-brancos (opcional, desligado por padrão): não grava tiles em que todos os pixels são brancos (255),
  como o fundo fora da área imageada; a grade e os nomes dos demais tiles não mudam.

Uso:
  python slicer.py imagem.tif --saida tiles512 [--prefixo JP] [--tamanho 512] [--compressao DEFLATE]
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()
BRANCO = 255


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", help="ortoimagem GeoTIFF")
    ap.add_argument("--saida", required=True, help="pasta de saída dos tiles")
    ap.add_argument("--tamanho", type=int, default=512)
    ap.add_argument("--prefixo", default=None, help="prefixo do nome dos tiles (padrão: nome da imagem)")
    ap.add_argument("--compressao", default="DEFLATE", choices=["DEFLATE", "LZW", "NONE"],
                    help="compressão sem perdas dos tiles (padrão DEFLATE)")
    ap.add_argument("--pular-brancos", action="store_true", help="não grava tiles 100%% brancos")
    a = ap.parse_args()

    T = a.tamanho
    ds = gdal.Open(a.entrada)
    W, H, B = ds.RasterXSize, ds.RasterYSize, ds.RasterCount
    gt = ds.GetGeoTransform()
    srs = ds.GetProjection()
    if not srs:
        sys.exit("a imagem não tem sistema de coordenadas: o georreferenciamento não pode ser preservado")
    tipo = ds.GetRasterBand(1).DataType
    if tipo != gdal.GDT_Byte:
        sys.exit(f"tipo de dado {gdal.GetDataTypeName(tipo)} não suportado: o branco (255) supõe 8 bits")
    cores = [ds.GetRasterBand(b + 1).GetColorInterpretation() for b in range(B)]
    nlin, ncol = math.ceil(H / T), math.ceil(W / T)
    dig = max(3, len(str(max(nlin, ncol))))
    prefixo = a.prefixo or Path(a.entrada).stem
    out = Path(a.saida)
    out.mkdir(parents=True, exist_ok=True)
    opcoes = ["TILED=NO"] + ([] if a.compressao == "NONE" else [f"COMPRESS={a.compressao}", "PREDICTOR=2"])
    drv = gdal.GetDriverByName("GTiff")
    print(f"imagem {W} x {H} px, {B} bandas, pixel {gt[1]:.4f} x {gt[5]:.4f}; grade {nlin} x {ncol} = "
          f"{nlin * ncol} tiles de {T} x {T}", flush=True)

    t0 = time.time()
    pulados = 0
    with open(out / "indice_tiles.csv", "w", newline="", encoding="utf-8") as fcsv:
        wcsv = csv.writer(fcsv)
        wcsv.writerow(["tile", "linha", "coluna", "x0_px", "y0_px", "largura_valida", "altura_valida",
                       "frac_completada", "x_origem", "y_origem"])
        for r in range(nlin):
            y0 = r * T
            h = min(T, H - y0)
            faixa = ds.ReadAsArray(0, y0, W, h)  # (B, h, W): uma linha de tiles por leitura
            if faixa.ndim == 2:
                faixa = faixa[None]
            for c in range(ncol):
                x0 = c * T
                w = min(T, W - x0)
                tile = np.full((B, T, T), BRANCO, dtype=np.uint8)
                tile[:, :h, :w] = faixa[:, :, x0:x0 + w]
                if a.pular_brancos and (tile == BRANCO).all():
                    pulados += 1
                    continue
                xo = gt[0] + x0 * gt[1] + y0 * gt[2]
                yo = gt[3] + x0 * gt[4] + y0 * gt[5]
                nome = f"{prefixo}_{r + 1:0{dig}d}_{c + 1:0{dig}d}"
                dst = drv.Create(str(out / f"{nome}.tif"), T, T, B, gdal.GDT_Byte, opcoes)
                dst.SetGeoTransform((xo, gt[1], gt[2], yo, gt[4], gt[5]))
                dst.SetProjection(srs)
                for b in range(B):
                    banda = dst.GetRasterBand(b + 1)
                    banda.WriteArray(tile[b])
                    banda.SetColorInterpretation(cores[b])
                dst.FlushCache()
                dst = None
                wcsv.writerow([nome, r + 1, c + 1, x0, y0, w, h, round(1 - (w * h) / (T * T), 6), xo, yo])
            print(f"linha {r + 1}/{nlin} ({time.time() - t0:.0f} s)", flush=True)
    print(f"concluído: {nlin * ncol - pulados} tiles gravados, {pulados} brancos pulados, em {out} "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
