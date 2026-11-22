#!/usr/bin/env python3
"""Cobertura de dossel da area urbana em grade de 100 m (2026-11-20).

Criterio fixado antes em runs/cobertura_dossel_20261120/criterio.md.

Etapas:
  validar  204 tiles de teste: cobertura predita (SegFormer MiT-B1, limiar congelado) x mascara de referencia,
           por tile e por grupo de tiles vizinhos; IC 95% por bootstrap por tile. Roda uma vez.
  cidade   tiles de tiles512 + novas_imagens que tocam o perimetro urbano: area de copa e area observada
           por celula de 100 m (SIRGAS 2000 / UTM 23S), so com pixels dentro do perimetro. Retomavel.
  mapa     figura da cobertura por celula (gerar depois de "cidade").

A segmentacao usa as mesmas funcoes do pipeline (pipeline_cidade.load_tile / pad_to / to_tensor), o mesmo
checkpoint e o mesmo limiar. Nao ha individualizacao: cobertura = mascara binaria.
Uso: python cobertura_dossel.py validar|cidade|mapa [--device cpu|cuda]
"""
import argparse
import json
import pickle
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import torch
from PIL import Image
from pyproj import Transformer
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform as shp_transform

import species_data as sd
import tree_georef as tg
from pipeline_cidade import TILE_SIZE, load_tile, pad_to, to_tensor

LAB = Path(__file__).resolve().parent
OUT = LAB / "runs" / "cobertura_dossel_20261120"
SEG_RUN = LAB / "runs" / "arq_segformer_mit_b1_20261110"
KML = LAB / "Placemark.kml"
TILE_DIRS = [Path("/home/paulomaciel/Datasets/drone_rp_v01/tiles512"), Path("/home/paulomaciel/proj/novas_imagens")]
TEST_IMG = Path("/home/paulomaciel/proj/imagens")
TEST_MSK = Path("/home/paulomaciel/proj/masks")
CELL = 100.0
MIN_AREA_PERIM = 2500.0
MIN_FRAC_OBS = 0.5
B, SEED = 2000, 42
TO_UTM = Transformer.from_crs(4326, 31983, always_xy=True).transform
TO_WGS = Transformer.from_crs(31983, 4326, always_xy=True).transform
CLASSES = [0, 5, 10, 20, 30, 101]  # limites das classes do mapa (%)


def load_model(device):
    import segmentation_models_pytorch as smp
    thr = json.loads((SEG_RUN / "frozen_threshold.json").read_text())
    if thr.get("criterion") != "validation_iou":
        raise ValueError(f"limiar nao escolhido na validacao: {thr}")
    seg = smp.Segformer(encoder_name="mit_b1", encoder_weights=None, in_channels=3, classes=1).to(device)
    seg.load_state_dict(torch.load(SEG_RUN / "checkpoint_best.pth", map_location=device, weights_only=True))
    seg.eval()
    return seg, float(thr["threshold"])


def predict(seg, device, loaded):
    x = torch.stack([to_tensor(pad_to(rgb, TILE_SIZE)) for rgb, _ in loaded]).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(seg(x))[:, 0].float().cpu().numpy()
    return [pr[:rgb.shape[0], :rgb.shape[1]] for (rgb, _), pr in zip(loaded, prob)]


def perimetro_utm():
    root = ET.parse(KML).getroot()
    txt = [e.text for e in root.iter() if e.tag.endswith("coordinates")]
    if len(txt) != 1:
        raise ValueError(f"{KML}: esperado 1 poligono, achei {len(txt)}")
    pts = [tuple(map(float, c.split(",")[:2])) for c in txt[0].split()]
    poly = shp_transform(TO_UTM, Polygon(pts))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def tile_coords(stem):
    m = re.search(r"_(\d{3})_(\d{3})$", stem)
    return int(m.group(1)), int(m.group(2))


def boot_ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ------------------------------------------------------------------ validar
def validar(device, batch):
    res_path = OUT / "validacao_teste.json"
    if res_path.exists():
        raise SystemExit("validacao ja feita (avaliacao unica): " + str(res_path))
    seg, thr = load_model(device)
    split, _ = sd.load_tile_split(sd.resolve_split_path())
    tiles = sorted(p for p in TEST_IMG.glob("*.tif") if split.get(p.stem) == "test")
    if len(tiles) != 204:
        raise ValueError(f"esperados 204 tiles de teste, achei {len(tiles)}")
    rows = []
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        loaded = [load_tile(p) for p in chunk]
        for p, (rgb, valid), pr in zip(chunk, loaded, predict(seg, device, loaded)):
            ref = np.any(np.asarray(Image.open(TEST_MSK / f"{p.stem}.png").convert("RGB")) != 0, axis=2)
            if ref.shape != valid.shape:
                raise ValueError(f"{p.stem}: mascara {ref.shape} x imagem {valid.shape}")
            pred = (pr >= thr) & valid
            ref = ref & valid
            r, c = tile_coords(p.stem)
            rows.append({"tile": p.stem, "linha": r, "coluna": c, "px_validos": int(valid.sum()),
                         "px_pred": int(pred.sum()), "px_ref": int(ref.sum()), "tp": int((pred & ref).sum())})
        print(f"{min(i + batch, len(tiles))}/{len(tiles)}", flush=True)
    d = pd.DataFrame(rows)
    d["cob_pred"] = 100 * d.px_pred / d.px_validos
    d["cob_ref"] = 100 * d.px_ref / d.px_validos
    d["erro_pp"] = d.cob_pred - d.cob_ref
    # grupos de tiles de teste vizinhos (vizinhanca-4)
    pos = {(r, c): k for k, (r, c) in enumerate(zip(d.linha, d.coluna))}
    comp = -np.ones(len(d), int)
    n = 0
    for k in range(len(d)):
        if comp[k] >= 0:
            continue
        pilha = [k]
        comp[k] = n
        while pilha:
            j = pilha.pop()
            r, c = d.linha[j], d.coluna[j]
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                q = pos.get(nb)
                if q is not None and comp[q] < 0:
                    comp[q] = n
                    pilha.append(q)
        n += 1
    d["grupo"] = comp
    d.to_csv(OUT / "validacao_por_tile.csv", index=False)
    g = d.groupby("grupo")[["px_validos", "px_pred", "px_ref"]].sum()
    g["n_tiles"] = d.groupby("grupo").size()
    g["cob_pred"] = 100 * g.px_pred / g.px_validos
    g["cob_ref"] = 100 * g.px_ref / g.px_validos
    g["erro_pp"] = g.cob_pred - g.cob_ref
    g.to_csv(OUT / "validacao_por_grupo.csv")

    def medidas(t):
        return {"vies_relativo": t.px_pred.sum() / t.px_ref.sum() - 1,
                "cob_pred_total": 100 * t.px_pred.sum() / t.px_validos.sum(),
                "cob_ref_total": 100 * t.px_ref.sum() / t.px_validos.sum()}

    rng = np.random.default_rng(SEED)
    vies, emed = [], []
    for _ in range(B):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        vies.append(s.px_pred.sum() / s.px_ref.sum() - 1)
        emed.append(s.erro_pp.mean())
    tp, fp, fn = d.tp.sum(), (d.px_pred - d.tp).sum(), (d.px_ref - d.tp).sum()
    res = {
        "n_tiles": len(d), "n_grupos": int(n), "limiar": thr,
        "iou_pixels_agregado_conferencia": tp / (tp + fp + fn),
        **medidas(d), "vies_relativo_ic95": boot_ci(vies),
        "por_tile": {"erro_medio_pp": d.erro_pp.mean(), "erro_medio_pp_ic95": boot_ci(emed),
                     "erro_absoluto_medio_pp": d.erro_pp.abs().mean(),
                     "pearson_r": float(np.corrcoef(d.cob_pred, d.cob_ref)[0, 1])},
        "por_grupo": {"erro_medio_pp": g.erro_pp.mean(), "erro_absoluto_medio_pp": g.erro_pp.abs().mean(),
                      "erro_absoluto_medio_pp_grupos_ge_9_tiles": g[g.n_tiles >= 9].erro_pp.abs().mean(),
                      "n_grupos_ge_9_tiles": int((g.n_tiles >= 9).sum()),
                      "pearson_r": float(np.corrcoef(g.cob_pred, g.cob_ref)[0, 1])},
        "criterio": "criterio.md",
    }
    res_path.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


# ------------------------------------------------------------------ cidade
def listar_tiles():
    tiles = {}
    for d in TILE_DIRS:  # tiles512 primeiro: os repetidos vem dele
        for p in sorted(d.glob("*.tif")):
            tiles.setdefault(p.stem, p)
    return [tiles[k] for k in sorted(tiles)]


def tile_geom(p, gt=None):
    gt = gt or tg.read_geotransform(p)
    w, h = Image.open(p).size
    lon0, lat0, sx, sy = gt
    cx = [lon0, lon0 + w * sx, lon0 + w * sx, lon0]
    cy = [lat0, lat0, lat0 - h * sy, lat0 - h * sy]
    e, n = TO_UTM(np.array(cx), np.array(cy))
    return Polygon(zip(e, n)), (e, n), (w, h)


def cidade(device, batch, chunk):
    per = perimetro_utm()
    shapely.prepare(per)
    parts = OUT / "partes"
    parts.mkdir(parents=True, exist_ok=True)
    lista = OUT / "tiles_perimetro.txt"
    if not lista.exists():
        t0 = time.perf_counter()
        sel = []
        for p in listar_tiles():
            poly, _, _ = tile_geom(p)
            if per.intersects(poly):
                sel.append(str(p))
        lista.write_text("\n".join(sel) + "\n")
        print(f"{len(sel)} tiles tocam o perimetro ({time.perf_counter() - t0:.0f} s)", flush=True)
    tiles = [Path(s) for s in lista.read_text().split()]
    seg, thr = load_model(device)
    chunks = [tiles[i:i + chunk] for i in range(0, len(tiles), chunk)]
    t0 = time.perf_counter()
    for ci, ch in enumerate(chunks):
        part = parts / f"parte_{ci:05d}.pkl"
        if part.exists():
            continue
        acc, nodata, n_borda = {}, 0, 0
        for bi in range(0, len(ch), batch):
            b = ch[bi:bi + batch]
            loaded = [load_tile(p) for p in b]
            for p, (rgb, valid), pr in zip(b, loaded, predict(seg, device, loaded)):
                if not valid.any():
                    nodata += 1
                    continue
                poly, (e, n), (w, h) = tile_geom(p)
                if (h, w) != valid.shape:
                    raise ValueError(f"{p}: tamanho inconsistente")
                # afim pixel -> UTM a partir de 3 cantos (tile de ~16 m: distorcao desprezivel)
                ax, ay = (e[1] - e[0]) / w, (n[1] - n[0]) / w
                bx, by = (e[3] - e[0]) / h, (n[3] - n[0]) / h
                area_px = abs(ax * by - ay * bx)
                yy, xx = np.mgrid[0:h, 0:w]
                E = e[0] + (xx + 0.5) * ax + (yy + 0.5) * bx
                N = n[0] + (xx + 0.5) * ay + (yy + 0.5) * by
                dentro = valid.copy()
                if not per.contains(poly):
                    n_borda += 1
                    dentro &= shapely.contains_xy(per, E, N)
                if not dentro.any():
                    continue
                copa = (pr >= thr) & dentro
                ix = np.floor(E[dentro] / CELL).astype(np.int64)
                iy = np.floor(N[dentro] / CELL).astype(np.int64)
                key = ix * 1_000_000 + iy
                uk, inv = np.unique(key, return_inverse=True)
                obs = np.bincount(inv) * area_px
                cop = np.bincount(inv, weights=copa[dentro].astype(np.float64)) * area_px
                for k, o, c in zip(uk.tolist(), obs, cop):
                    a = acc.setdefault(k, [0.0, 0.0])
                    a[0] += o
                    a[1] += c
        with open(part, "wb") as fh:
            pickle.dump({"acc": acc, "nodata": nodata, "borda": n_borda, "tiles": [p.stem for p in ch]}, fh)
        feitos = sum(len(c) for c in chunks[:ci + 1])
        dt = time.perf_counter() - t0
        print(f"parte {ci + 1}/{len(chunks)}: {feitos}/{len(tiles)} tiles ({dt / 60:.1f} min)", flush=True)
    consolidar(per, thr, len(tiles))


def consolidar(per, thr, n_tiles):
    acc, nodata, borda = {}, 0, 0
    for part in sorted((OUT / "partes").glob("parte_*.pkl")):
        d = pickle.load(open(part, "rb"))
        nodata += d["nodata"]
        borda += d["borda"]
        for k, (o, c) in d["acc"].items():
            a = acc.setdefault(k, [0.0, 0.0])
            a[0] += o
            a[1] += c
    rows, feats = [], []
    # todas as celulas que tocam o perimetro (inclusive as sem nenhum pixel observado)
    x0, y0, x1, y1 = per.bounds
    for ix in range(int(np.floor(x0 / CELL)), int(np.floor(x1 / CELL)) + 1):
        for iy in range(int(np.floor(y0 / CELL)), int(np.floor(y1 / CELL)) + 1):
            cel = box(ix * CELL, iy * CELL, (ix + 1) * CELL, (iy + 1) * CELL)
            parte = cel.intersection(per)
            if parte.is_empty:
                continue
            o, c = acc.get(ix * 1_000_000 + iy, (0.0, 0.0))
            ap = parte.area
            exib = ap >= MIN_AREA_PERIM and o >= MIN_FRAC_OBS * ap
            cob = 100 * c / o if o > 0 else np.nan
            rows.append({"ix": ix, "iy": iy, "area_perimetro_m2": ap, "area_observada_m2": o, "area_copa_m2": c,
                         "frac_observada": o / ap, "cobertura_pct": cob, "exibida": bool(exib)})
            feats.append({"type": "Feature", "geometry": mapping(shp_transform(TO_WGS, parte)),
                          "properties": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                                         for k, v in rows[-1].items()}})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "celulas.csv", index=False)
    (OUT / "celulas.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    ex = df[df.exibida]
    q = ex.cobertura_pct.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    resumo = {
        "n_tiles_perimetro": n_tiles, "n_tiles_sem_dado": nodata, "n_tiles_borda_perimetro": borda,
        "limiar": thr, "celula_m": CELL, "area_perimetro_ha": per.area / 1e4,
        "area_observada_ha": df.area_observada_m2.sum() / 1e4, "area_copa_ha": df.area_copa_m2.sum() / 1e4,
        "cobertura_area_observada_pct": 100 * df.area_copa_m2.sum() / df.area_observada_m2.sum(),
        "n_celulas": len(df), "n_celulas_exibidas": int(len(ex)),
        "cobertura_celulas_quantis": {f"p{int(k * 100)}": float(v) for k, v in q.items()},
        "celulas_por_classe": {f"{a}-{b if b <= 100 else 100}": int(((ex.cobertura_pct >= a) &
                                                                    (ex.cobertura_pct < b)).sum())
                               for a, b in zip(CLASSES[:-1], CLASSES[1:])},
    }
    (OUT / "resumo.json").write_text(json.dumps(resumo, indent=2, default=float))
    print(json.dumps(resumo, indent=2, default=float))


# ------------------------------------------------------------------ mapa
def mapa(saida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    df = pd.read_csv(OUT / "celulas.csv")
    per = perimetro_utm()
    cores = ["#c8553d", "#e9a97a", "#f3e3b3", "#8fc98a", "#2f7d4a"]
    rot = ["< 5%", "5–10%", "10–20%", "20–30%", "≥ 30%"]
    cmap, norm = ListedColormap(cores), BoundaryNorm(CLASSES, len(cores))
    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    for r in df.itertuples():
        cel = box(r.ix * CELL, r.iy * CELL, (r.ix + 1) * CELL, (r.iy + 1) * CELL).intersection(per)
        geoms = getattr(cel, "geoms", [cel])
        for gm in geoms:
            if gm.geom_type != "Polygon":
                continue
            x, y = gm.exterior.xy
            if r.exibida:
                ax.fill(x, y, color=cmap(norm(r.cobertura_pct)), ec="white", lw=0.4)
            else:
                ax.fill(x, y, color="#d9d9d9", ec="white", lw=0.4, hatch="////")
    x, y = per.exterior.xy
    ax.plot(x, y, color="#333333", lw=1.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    x0, y0, x1, y1 = per.bounds
    L = 500
    ax.plot([x0, x0 + L], [y0 - 60, y0 - 60], color="black", lw=3)
    ax.text(x0 + L / 2, y0 - 40, f"{L} m", ha="center", va="bottom", fontsize=11)
    ax.annotate("N", xy=(x1, y1), xytext=(x1, y1 - 250), ha="center", fontsize=13, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    hs = [Patch(color=c, label=l) for c, l in zip(cores, rot)] + [
        Patch(facecolor="#d9d9d9", hatch="////", edgecolor="white", label="Borda (não avaliada)")]
    ax.legend(handles=hs, title="Cobertura de copa", loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=10, title_fontsize=11,
              frameon=True)
    fig.savefig(saida, bbox_inches="tight", facecolor="white")
    print("mapa:", saida)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=["validar", "cidade", "mapa"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--saida", default=str(OUT / "mapa_cobertura_dossel.png"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.etapa == "validar":
        validar(torch.device(a.device), a.batch)
    elif a.etapa == "cidade":
        cidade(torch.device(a.device), a.batch, a.chunk)
    else:
        mapa(a.saida)


if __name__ == "__main__":
    main()
