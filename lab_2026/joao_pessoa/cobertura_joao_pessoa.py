#!/usr/bin/env python3
"""Cobertura de copa em células de 100 m em João Pessoa (mapa de calor), 2026-11-22.

Mesmas faixas e regra de borda do mapa de Rio Paranaíba (lab_2026/cobertura_dossel.py), com duas diferenças:
  - a copa vem dos polígonos das copas do modelo do sistema (resultados/joao_pessoa/atual/copas.geojson),
    rasterizados a 0,5 m, e não da máscara da segmentação (em Rio Paranaíba, as duas medidas tiveram
    correlação de 0,94 por célula);
  - a área de referência de cada célula é a área imageada (pixels não brancos da ortoimagem), e não um
    perímetro urbano: células com menos de 2.500 m² imageados são bordas e não são avaliadas.
Grade alinhada a múltiplos de 100 m em SIRGAS 2000 / UTM 25S (EPSG:31985).
Saídas em resultados/joao_pessoa/cobertura/: celulas.geojson, resumo.json, mapa_cobertura.png.
Uso (ambiente conda "geo"): python cobertura_joao_pessoa.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from osgeo import gdal, osr  # noqa: E402

gdal.UseExceptions()
BASE = Path(__file__).resolve().parent
SRC = BASE / "6a90442d7efa87881ba82682.tif"
COPAS = BASE / "resultados" / "joao_pessoa" / "atual" / "copas.geojson"
OUT = BASE / "resultados" / "joao_pessoa" / "cobertura"
RES, CEL, MIN_OBS = 0.5, 100.0, 2500.0  # m: resolução da grade fina, lado da célula, área mínima avaliada
CLASSES = [0, 5, 10, 20, 30, 101]
CORES = ["#c8553d", "#e9a97a", "#f3e3b3", "#8fc98a", "#2f7d4a"]
ROTULOS = ["< 5%", "5–10%", "10–20%", "20–30%", "≥ 30%"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 1. área imageada a 0,5 m (pixel não branco e não preto, com tolerância de 10 níveis)
    ds = gdal.Translate("", str(SRC), format="MEM", xRes=RES, yRes=RES, resampleAlg="nearest")
    gt, srs = ds.GetGeoTransform(), ds.GetProjection()
    a = ds.ReadAsArray()
    valido = ~((a >= 245).all(0) | (a <= 10).all(0))  # quase branco ou quase preto (borda JPEG) = fora
    H, W = valido.shape
    # 2. copas rasterizadas na mesma grade (reprojetadas de WGS 84 para UTM 25S antes)
    utm_json = OUT / "copas_utm25s.geojson"
    gdal.VectorTranslate(str(utm_json), str(COPAS), dstSRS="EPSG:31985", format="GeoJSON")
    m = gdal.GetDriverByName("MEM").Create("", W, H, 1, gdal.GDT_Byte)
    m.SetGeoTransform(gt)
    m.SetProjection(srs)
    gdal.Rasterize(m, str(utm_json), burnValues=[1])
    copa = m.ReadAsArray().astype(bool) & valido
    utm_json.unlink()
    # 3. células de 100 m alinhadas a múltiplos de 100 m
    x0, y0 = gt[0], gt[3]
    xs = np.floor(x0 / CEL) * CEL
    ys = np.ceil(y0 / CEL) * CEL
    ncol = int(np.ceil((x0 + W * RES - xs) / CEL))
    nlin = int(np.ceil((ys - (y0 - H * RES)) / CEL))
    # coluna/linha de célula de cada pixel fino
    cx = ((x0 + (np.arange(W) + 0.5) * RES - xs) // CEL).astype(int)
    cy = ((ys - (y0 - (np.arange(H) + 0.5) * RES)) // CEL).astype(int)
    idx = cy[:, None] * ncol + cx[None, :]
    obs = np.bincount(idx.ravel(), weights=valido.ravel(), minlength=nlin * ncol) * RES * RES
    cop = np.bincount(idx.ravel(), weights=copa.ravel(), minlength=nlin * ncol) * RES * RES
    to_wgs = osr.CoordinateTransformation(
        osr.SpatialReference(wkt=srs), (lambda s: (s.ImportFromEPSG(4326), s)[1])(osr.SpatialReference()))
    feats, cels, obs_cel = [], [], {}
    for k in np.nonzero(obs > 0)[0]:
        i, j = divmod(int(k), ncol)
        xa, yb = xs + j * CEL, ys - i * CEL
        pct = 100 * cop[k] / obs[k]
        exib = bool(obs[k] >= MIN_OBS)
        anel = [(xa, yb), (xa + CEL, yb), (xa + CEL, yb - CEL), (xa, yb - CEL), (xa, yb)]
        ll = [list(to_wgs.TransformPoint(x, y)[:2])[::-1] for x, y in anel]  # (lon, lat)
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ll]},
                      "properties": {"linha": i, "coluna": j, "x_utm": xa, "y_utm": yb,
                                     "area_obs_m2": round(float(obs[k]), 1), "area_copa_m2": round(float(cop[k]), 1),
                                     "cobertura_pct": round(float(pct), 2), "exibida": exib}})
        cels.append((xa, yb, pct, exib))
        obs_cel[(xa, yb)] = float(obs[k])
    (OUT / "celulas.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    ex = np.array([c[2] for c in cels if c[3]])
    resumo = {"area_imageada_ha": round(float(obs.sum()) / 1e4, 2), "area_copa_ha": round(float(cop.sum()) / 1e4, 2),
              "cobertura_pct": round(100 * float(cop.sum() / obs.sum()), 2), "celulas_total": len(cels),
              "celulas_avaliadas": int(len(ex)), "celulas_borda": int(len(cels) - len(ex)),
              "mediana_pct": round(float(np.median(ex)), 2), "p90_pct": round(float(np.percentile(ex, 90)), 2),
              "celulas_abaixo_5pct": int((ex < 5).sum()), "celulas_30pct_ou_mais": int((ex >= 30).sum()),
              "por_faixa": {r: int(((ex >= lo) & (ex < hi)).sum()) for r, lo, hi in zip(ROTULOS, CLASSES[:-1], CLASSES[1:])},
              "fonte_copa": "copas.geojson do modelo do sistema (SegFormer 3 cm + h-maxima), rasterizadas a 0,5 m",
              "observacao": "inclui as detecções falsas (precisão de 92,9% na conferência de 100 árvores; 55 sobre a lagoa)"}
    (OUT / "resumo.json").write_text(json.dumps(resumo, indent=1, ensure_ascii=False))
    print(json.dumps(resumo, indent=1, ensure_ascii=False))

    # 4. mapa sobre a ortoimagem
    img_ds = gdal.Translate("", str(SRC), format="MEM", width=3000)
    img = np.moveaxis(img_ds.ReadAsArray(), 0, -1)
    g2 = img_ds.GetGeoTransform()
    vis = img.copy()
    cmap, norm = ListedColormap(CORES), BoundaryNorm(CLASSES, len(CORES))
    fig, ax = plt.subplots(figsize=(12, 12 * img.shape[0] / img.shape[1]))
    ax.imshow(vis)
    for xa, yb, pct, exib in cels:
        px, py = (xa - g2[0]) / g2[1], (yb - g2[3]) / g2[5]
        lado = CEL / g2[1]
        if exib:
            ax.add_patch(Rectangle((px, py), lado, lado, fc=cmap(norm(pct)), alpha=0.6, ec="white", lw=0.4))
        elif obs_cel[(xa, yb)] >= 500:  # células com < 500 m² são a moldura cinza do arquivo original: não desenha
            ax.add_patch(Rectangle((px, py), lado, lado, fc="none", ec="#777777", lw=0.4, hatch="////"))
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    m500 = 500 / g2[1]
    ax.plot([60, 60 + m500], [img.shape[0] - 60] * 2, color="black", lw=4)
    ax.text(60 + m500 / 2, img.shape[0] - 80, "500 m", ha="center", fontsize=12)
    hs = [Patch(fc=c, alpha=0.6, ec="#555555", label=r) for c, r in zip(CORES, ROTULOS)] + [
        Patch(fc="white", hatch="////", ec="#777777", label="Borda (não avaliada)")]
    ax.legend(handles=hs, title="Cobertura de copa", loc="upper right", fontsize=11, title_fontsize=12)
    ax.set_axis_off()
    ax.set_title(f"João Pessoa: cobertura de copa por célula de 100 m ({resumo['cobertura_pct']:.1f}% "
                 "da área imageada)".replace(".", ","), fontsize=14)
    fig.savefig(OUT / "mapa_cobertura.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saída:", OUT)


if __name__ == "__main__":
    main()
