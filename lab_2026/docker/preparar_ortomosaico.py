#!/usr/bin/env python3
"""Prepara um ortomosaico RGB GeoTIFF para o pipeline SIACE (generalização de preparar_jp_3cm.py, 2026-11-22).

O sistema foi treinado em tiles de 512 x 512 px com pixel de 3,19 cm em EPSG:4326. A imagem é:
  1. reduzida às bandas 1-3 (RGB, 8 bits);
  2. reprojetada para EPSG:4326 com pixel equivalente a 3,19 cm no terreno (bilinear; fundo branco 255);
  3. recortada por slicer.py: 512 x 512, sem sobreposição, bordas brancas, tiles 100% brancos pulados.
Imagens de resolução pior que 3,19 cm são ampliadas para essa grade (o detalhe não aumenta).
Também escolhe a zona UTM SIRGAS 2000 (Brasil, zonas 18 a 25) ou WGS 84 pelo centro da imagem, usada no
cálculo de áreas, e grava <saida>/preparo.json.
Uso: python3 preparar_ortomosaico.py orto.tif --saida pasta [--prefixo ORTO]
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from osgeo import gdal
from pyproj import Geod

gdal.UseExceptions()
PIXEL_M = 0.0319
AQUI = Path(__file__).resolve().parent


def epsg_utm(lon, lat):
    zona = int(math.floor((lon + 180) / 6)) + 1
    if lat < 0 and 18 <= zona <= 25 and -35 <= lat and -75 <= lon <= -28:
        return 31960 + zona  # SIRGAS 2000 / UTM zona S (ex.: 31983 = 23S)
    return (32700 if lat < 0 else 32600) + zona  # WGS 84 / UTM


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--prefixo", default="ORTO")
    a = ap.parse_args()
    out = Path(a.saida)
    out.mkdir(parents=True, exist_ok=True)
    ds = gdal.Open(a.entrada)
    if ds.RasterCount < 3 or ds.GetRasterBand(1).DataType != gdal.GDT_Byte:
        sys.exit("a imagem precisa ter pelo menos 3 bandas de 8 bits (RGB)")
    if not ds.GetProjection():
        sys.exit("a imagem não tem sistema de coordenadas")
    rgb = out / "rgb.vrt"
    gdal.Translate(str(rgb), ds, format="VRT", bandList=[1, 2, 3])
    cs = gdal.Info(str(rgb), format="json")["wgs84Extent"]["coordinates"][0][:4]
    lon_c, lat_c = sum(c[0] for c in cs) / 4, sum(c[1] for c in cs) / 4
    g = Geod(ellps="WGS84")
    m_lon = g.inv(lon_c - 0.0005, lat_c, lon_c + 0.0005, lat_c)[2] / 0.001
    m_lat = g.inv(lon_c, lat_c - 0.0005, lon_c, lat_c + 0.0005)[2] / 0.001
    xres, yres = PIXEL_M / m_lon, PIXEL_M / m_lat
    vrt = out / "orto_4326_3cm.vrt"
    gdal.Warp(str(vrt), str(rgb), format="VRT", dstSRS="EPSG:4326", xRes=xres, yRes=yres, resampleAlg="bilinear",
              srcNodata="None", dstNodata="None", warpOptions=["INIT_DEST=255"], multithread=True)
    gt_src = ds.GetGeoTransform()
    info = {"entrada": str(a.entrada), "centro_lon": lon_c, "centro_lat": lat_c, "utm_epsg": epsg_utm(lon_c, lat_c),
            "pixel_original": [gt_src[1], gt_src[5]], "pixel_grade_graus": [xres, -yres], "pixel_grade_m": PIXEL_M}
    (out / "preparo.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    print(json.dumps(info, ensure_ascii=False), flush=True)
    subprocess.run([sys.executable, str(AQUI / "slicer.py"), str(vrt), "--saida", str(out / "tiles"),
                    "--prefixo", a.prefixo, "--pular-brancos"], check=True)


if __name__ == "__main__":
    main()
