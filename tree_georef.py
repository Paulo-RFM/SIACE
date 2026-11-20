#!/usr/bin/env python3
"""Georreferencia as copas individuais (saida do watershed) e salva cada arvore.

Para cada tile:
  1. le a georreferencia das tags GeoTIFF do proprio tile (sem rasterio/GDAL):
     ModelTiepoint (33922) = coordenada do canto superior esquerdo,
     ModelPixelScale (33550) = tamanho do pixel em graus, GeoKey 2048 = EPSG:4326;
  2. converte o contorno de cada instancia: lon = lon0 + x*sx ; lat = lat0 - y*sy;
  3. calcula centroide (centro de massa dos pixels; se cair fora da copa usa
     representative_point) e area em m2 (poligono projetado em SIRGAS 2000 / UTM 23S);
  4. junta copas cortadas pela borda entre tiles vizinhos (poligonos que se tocam,
     ambos encostados na borda do tile);
  5. grava GeoJSON (copas e pontos, WGS 84) + CSV (lat/lon) + resumo JSON.

Uso:
  python tree_georef.py --instances-dir <dir com *_instances.png> --images-dir /home/paulomaciel/proj/imagens \
      --output-dir <saida> [--classes-csv predicoes.csv]
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Point, Polygon, mapping
from shapely.ops import transform as shp_transform, unary_union
from shapely.strtree import STRtree

SIMPLIFY_TOLERANCE_PX = 1.0
UTM_EPSG = 31983  # SIRGAS 2000 / UTM zona 23S (Rio Paranaiba-MG)
TO_UTM = Transformer.from_crs(4326, UTM_EPSG, always_xy=True).transform
TO_WGS = Transformer.from_crs(UTM_EPSG, 4326, always_xy=True).transform


def read_geotransform(tif_path):
    """Retorna (lon0, lat0, sx, sy) a partir das tags GeoTIFF; falha se nao for EPSG:4326."""
    tags = Image.open(tif_path).tag_v2
    if 33922 not in tags or 33550 not in tags:
        raise ValueError(f"{tif_path}: sem ModelTiepoint/ModelPixelScale (nao georreferenciado).")
    tp, sc = tags[33922], tags[33550]
    if tp[0] != 0 or tp[1] != 0:
        raise ValueError(f"{tif_path}: tiepoint nao esta no pixel (0,0): {tp[:2]}")
    geokeys = list(tags.get(34735, ()))
    keys = {geokeys[i]: geokeys[i + 3] for i in range(4, len(geokeys), 4)}
    if keys.get(2048) != 4326:
        raise ValueError(f"{tif_path}: GeographicTypeGeoKey={keys.get(2048)}, esperado 4326 (WGS 84).")
    # A formula lon = lon0 + x*sx so vale para grade alinhada ao norte, em graus, com o tiepoint no
    # CANTO do pixel (0,0). Qualquer outro caso daria coordenadas erradas sem aviso: falha explicita.
    if 34264 in tags:
        raise ValueError(f"{tif_path}: possui ModelTransformation (rotacao/cisalhamento); nao suportado.")
    if keys.get(1025, 1) != 1:
        raise ValueError(f"{tif_path}: GTRasterTypeGeoKey={keys.get(1025)} (PixelIsPoint); esperado PixelIsArea (1).")
    if keys.get(2054, 9102) != 9102:
        raise ValueError(f"{tif_path}: unidade angular {keys.get(2054)}; esperado graus (9102).")
    if not (sc[0] > 0 and sc[1] > 0):
        raise ValueError(f"{tif_path}: escala de pixel invalida {sc[:2]}.")
    return float(tp[3]), float(tp[4]), float(sc[0]), float(sc[1])


def pixel_to_lonlat(x, y, gt):
    lon0, lat0, sx, sy = gt
    return lon0 + x * sx, lat0 - y * sy


def instance_polygon(mask):
    """Contorno externo da instancia em coordenadas de pixel (bordas dos pixels)."""
    up = cv2.resize(mask.astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) < 3:
            continue
        pts = c[:, 0, :].astype(float) / 2.0  # volta para a escala original (bordas dos pixels)
        p = Polygon(pts).buffer(0)
        if not p.is_empty:
            polys.append(p)
    if not polys:
        return None
    poly = max(polys, key=lambda p: p.area) if len(polys) > 1 else polys[0]
    # Simplifica com tolerancia de 1 pixel (~3 cm): remove degraus da grade sem alterar a forma.
    return poly.simplify(SIMPLIFY_TOLERANCE_PX, preserve_topology=True)


def tile_records(inst, tile, gt, size):
    recs = []
    h, w = inst.shape
    for inst_id in [int(i) for i in np.unique(inst) if i > 0]:
        m = inst == inst_id
        poly_px = instance_polygon(m)
        if poly_px is None:
            continue
        poly_ll = shp_transform(lambda x, y, z=None: pixel_to_lonlat(np.asarray(x), np.asarray(y), gt), poly_px)
        ys, xs = np.nonzero(m)
        c_ll = Point(*pixel_to_lonlat(xs.mean() + 0.5, ys.mean() + 0.5, gt))
        if not poly_ll.contains(c_ll):
            c_ll = poly_ll.representative_point()
        touches = bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
        recs.append({"tile": tile, "instance_id": inst_id, "n_pixels": int(m.sum()),
                     "touches_tile_border": touches, "geometry": poly_ll, "centroid": c_ll})
    return recs


MERGE_RULE = "uma_parte_por_tile"


def merge_border_crowns(df, tol_m=0.10, rule=MERGE_RULE):
    """Une copas encostadas na borda que se tocam entre tiles diferentes (mesma arvore cortada).

    rule="uma_parte_por_tile" (padrao desde 2026-11-04): os pares candidatos sao unidos em ordem
    decrescente de comprimento de contato na emenda, e uma uniao so e aceita se os dois grupos nao
    tiverem nenhum tile em comum. Assim um grupo tem no maximo uma parte de cada tile: duas copas
    que a individualizacao ja separou dentro de um tile nao voltam a ser fundidas por encadeamento
    atraves do tile vizinho (na cidade, a regra antiga gerou um grupo de 803 partes).
    rule="encadeada": regra original (uniao transitiva de todo par que se toca), mantida para reproducao.
    """
    if rule not in ("uma_parte_por_tile", "encadeada"):
        raise ValueError(f"regra de uniao desconhecida: {rule}")
    df = df.reset_index(drop=True)
    utm = [shp_transform(TO_UTM, g) for g in df["geometry"]]
    border = df.index[df["touches_tile_border"]].tolist()
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    # Indice espacial (STRtree): compara so copas vizinhas; O(n log n) em vez de O(n^2).
    buffered = [utm[i].buffer(tol_m) for i in border]
    tree = STRtree(buffered)
    tiles = df["tile"].to_numpy()
    pairs = []
    for a_pos, a in enumerate(border):
        for b_pos in tree.query(buffered[a_pos], predicate="intersects"):
            b = border[int(b_pos)]
            if b > a and tiles[a] != tiles[b]:
                # contato ~ area da faixa comum / largura da faixa (2*tol)
                contact = buffered[a_pos].intersection(buffered[int(b_pos)]).area / (2 * tol_m)
                pairs.append((contact, a, b))
    if rule == "encadeada":
        for _, a, b in pairs:
            parent[find(a)] = find(b)
    else:
        tile_sets = {i: {tiles[i]} for i in border}
        for _, a, b in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
            ra, rb = find(a), find(b)
            if ra == rb or tile_sets[ra] & tile_sets[rb]:
                continue
            parent[ra] = rb
            tile_sets[rb] |= tile_sets.pop(ra)
    groups = {}
    for i in range(len(df)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for members in groups.values():
        # Costura a fresta (~1 cm) da emenda entre tiles: expande tol, une, encolhe tol.
        # Nunca descarta partes; a area e a soma exata das partes (tiles nao se sobrepoem).
        geom_utm = unary_union([utm[i].buffer(tol_m) for i in members]).buffer(-tol_m)
        area_m2 = float(sum(utm[i].area for i in members))
        base = df.loc[members].sort_values("n_pixels", ascending=False).iloc[0]
        # Centro de massa das partes (ponderado por pixels), mesma regra das copas de um so tile.
        w = df.loc[members, "n_pixels"].to_numpy(dtype=float)
        pts = [shp_transform(TO_UTM, df.at[i, "centroid"]) for i in members]
        cen = Point(float(np.average([q.x for q in pts], weights=w)), float(np.average([q.y for q in pts], weights=w)))
        if not geom_utm.contains(cen):
            cen = geom_utm.representative_point()
        out.append({
            "tile": base["tile"], "instance_id": int(base["instance_id"]),
            "parts": ";".join(f"{df.at[i, 'tile']}#{int(df.at[i, 'instance_id'])}" for i in members),
            "n_parts": len(members), "n_pixels": int(df.loc[members, "n_pixels"].sum()),
            "area_m2": area_m2,
            "geometry": shp_transform(TO_WGS, geom_utm),
            "centroid": shp_transform(TO_WGS, cen) if len(members) > 1 else base["centroid"],
        })
    return pd.DataFrame(out)


def write_geojson(path, records, geom_key, props):
    feats = [{"type": "Feature", "geometry": mapping(r[geom_key]),
              "properties": {k: (None if pd.isna(r[k]) else r[k]) for k in props}} for _, r in records.iterrows()]
    Path(path).write_text(json.dumps({"type": "FeatureCollection", "name": Path(path).stem,
                                      "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                                      "features": feats}, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--classes-csv", default=None,
                    help="CSV opcional com tile,instance_id,predicted_class,predicted_name,confidence")
    ap.add_argument("--no-merge", action="store_true", help="nao unir copas cortadas pela borda")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.instances_dir).glob("*_instances.png"))
    if not files:
        raise FileNotFoundError(f"Nenhum *_instances.png em {args.instances_dir}")
    recs, gsd = [], []
    for f in files:
        tile = f.name[: -len("_instances.png")]
        tif = Path(args.images_dir) / f"{tile}.tif"
        gt = read_geotransform(tif)
        inst = np.array(Image.open(f)).astype(np.int32)
        lat_c = gt[1] - gt[3] * inst.shape[0] / 2
        gsd.append((gt[2] * 111320 * np.cos(np.radians(abs(lat_c))), gt[3] * 110574))
        recs += tile_records(inst, tile, gt, inst.shape)
    df = pd.DataFrame(recs)
    n_raw = len(df)
    if df.empty:
        raise RuntimeError("Nenhuma instancia encontrada.")
    if args.no_merge:
        df["parts"] = df["tile"] + "#" + df["instance_id"].astype(str)
        df["n_parts"] = 1
        df["area_m2"] = [shp_transform(TO_UTM, g).area for g in df["geometry"]]
        df = df.drop(columns=["touches_tile_border"])
    else:
        df = merge_border_crowns(df)

    if args.classes_csv:
        cls = pd.read_csv(args.classes_csv)
        df = df.merge(cls, on=["tile", "instance_id"], how="left", validate="one_to_one")
    df.insert(0, "tree_id", range(1, len(df) + 1))
    df["lon"] = [c.x for c in df["centroid"]]
    df["lat"] = [c.y for c in df["centroid"]]
    props = [c for c in df.columns if c not in ("geometry", "centroid")]
    write_geojson(out / "copas.geojson", df, "geometry", props)
    write_geojson(out / "arvores.geojson", df, "centroid", props)
    df[props].to_csv(out / "arvores.csv", index=False)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(), "script": Path(__file__).name,
        "instances_dir": str(args.instances_dir), "images_dir": str(args.images_dir),
        "n_tiles": len(files), "n_instances_raw": n_raw, "n_trees": int(len(df)),
        "n_merged_across_tiles": int((df["n_parts"] > 1).sum()),
        "crs_output": "EPSG:4326 (WGS 84)", "area_crs": f"EPSG:{UTM_EPSG}",
        "gsd_m_mean": [float(np.mean([g[0] for g in gsd])), float(np.mean([g[1] for g in gsd]))],
        "area_m2_median": float(df["area_m2"].median()),
        "note": "centroide da copa vista de cima, nao a posicao do tronco; precisao absoluta depende do GPS/ortofoto",
    }
    (out / "georef_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
