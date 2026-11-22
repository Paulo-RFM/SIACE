#!/usr/bin/env python3
"""Prepara a ortoimagem de João Pessoa para o sistema SIACE (experimento de resolução, 2026-11-21).

O sistema foi treinado em tiles de 512 x 512 px com pixel de 3,19 cm, em EPSG:4326. Aqui a imagem
(8 cm, EPSG:31985) é reprojetada para EPSG:4326 com pixel equivalente a 3,19 cm no terreno, por interpolação
bilinear (a mesma usada para voltar à grade de 3,19 cm nos conjuntos de treino degradados), e recortada pelo
slicer.py: 512 x 512, sem sobreposição, bordas brancas, tiles 100% brancos pulados.
A imagem continua com o detalhe de 8 cm; só a grade muda. Fundo fora da área reprojetada = branco (255).
Uso: python preparar_jp_3cm.py
"""
import math
import subprocess
import sys
from pathlib import Path

from osgeo import gdal
from pyproj import Geod

gdal.UseExceptions()
BASE = Path(__file__).resolve().parent
SRC = BASE / "6a90442d7efa87881ba82682.tif"
VRT = BASE / "jp_4326_3cm.vrt"
PIXEL_M = 0.0319

ds = gdal.Open(str(SRC))
info = gdal.Info(ds, format="json")
lon_c, lat_c = info["wgs84Extent"]["coordinates"][0][0]  # canto; o centro é calculado abaixo
cs = info["wgs84Extent"]["coordinates"][0]
lon_c = sum(c[0] for c in cs[:4]) / 4
lat_c = sum(c[1] for c in cs[:4]) / 4
g = Geod(ellps="WGS84")
m_lon = g.inv(lon_c - 0.0005, lat_c, lon_c + 0.0005, lat_c)[2] / 0.001
m_lat = g.inv(lon_c, lat_c - 0.0005, lon_c, lat_c + 0.0005)[2] / 0.001
xres, yres = PIXEL_M / m_lon, PIXEL_M / m_lat
print(f"centro {lon_c:.5f}, {lat_c:.5f}; pixel {xres:.4e} x {yres:.4e} graus (= {PIXEL_M} m)")
gdal.Warp(str(VRT), ds, format="VRT", dstSRS="EPSG:4326", xRes=xres, yRes=yres, resampleAlg="bilinear",
          srcNodata="None", dstNodata="None", warpOptions=["INIT_DEST=255"], multithread=True)
subprocess.run([sys.executable, str(BASE / "slicer.py"), str(VRT), "--saida", str(BASE / "tiles512_3cm_4326"),
                "--prefixo", "JP3", "--pular-brancos"], check=True)
