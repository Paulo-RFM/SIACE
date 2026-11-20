#!/usr/bin/env python3
"""Calibra NA VALIDACAO os filtros para sugerir arvores ao OpenStreetMap e mede o resultado UMA vez no teste.

Entradas: runs do pipeline com --save-masks e o campo seg_prob_mean (pipeline_cidade.py >= 2026-11-04)
sobre as particoes de validacao e de teste, e os poligonos anotados (labels/ e labels2/).

Rotulo de cada copa detectada (por tile, antes da uniao entre tiles), pensado para o uso no OSM, em que a
arvore vira UM PONTO: a copa "e arvore" se o seu ponto (centroide, ou ponto interno se o centroide cair
fora) estiver dentro de um poligono de arvore anotado; a especie "verdadeira" e a desse poligono.

Filtros e criterios (todos definidos antes de olhar o teste):
  1. area maxima da copa = percentil 99 da area dos poligonos anotados do TREINO (exclui macicos de
     vegetacao, que nao sao arvores isoladas);
  2. limiar de certeza de arvore s (seg_prob_mean): o menor s da grade cuja precisao "e arvore" na
     validacao seja >= --precisao-arvore (padrao 0,95);
  3. especie: so e indicada se a confianca do classificador >= --conf-especie (padrao 0,70, sugerido pelo
     autor) E se, na validacao, a precisao da classe com esse limiar for >= --precisao-especie (padrao 0,80)
     com pelo menos --min-suporte (padrao 10) copas; classes que nao atingem isso nunca recebem especie.
     A precisao da especie e medida em TODAS as copas da validacao que estao sobre arvore (apos o filtro de
     area), e nao so nas aprovadas pelo limiar de arvore: as duas decisoes sao separadas, e o limiar de
     arvore muda a mistura de especies (v2, 2026-11-05; a v1 media so nas aprovadas e ficava sem suporte).
     As classes 1 (sem identificacao) e 3 (arvore seca) nunca recebem especie.
Saida: runs/osm_limiares_<data>/{limiares_osm.json, validacao_curva_arvore.csv, validacao_especies.csv,
       teste_resultado.json, copas_val.csv, copas_teste.csv}. Rodar a partir de lab_2026/.
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import Point, Polygon

LAB = Path(__file__).resolve().parent
P = Path("/home/paulomaciel/proj")
PIXEL_M = 0.0319
NO_SPECIES = {1, 3}


def polygons_of(tile):
    jp = next((q for q in [P / "labels" / f"{tile}.json", P / "labels2" / f"{tile}.json"] if q.exists()), None)
    out = []
    if jp:
        for sh in json.load(open(jp))["shapes"]:
            if sh.get("label") != "tree" or len(sh["points"]) < 3:
                continue
            poly = Polygon(sh["points"]).buffer(0)
            gid = sh.get("group_id")
            out.append((poly, int(gid) if gid is not None else 1))
    return out


def label_instances(run_dir):
    inst = pd.read_csv(run_dir / "instancias.csv")
    if inst["seg_prob_mean"].isna().any():
        raise SystemExit(f"{run_dir}: instancias sem seg_prob_mean; rode o pipeline atualizado.")
    rows = []
    for tile, g in inst.groupby("tile"):
        lab = np.array(Image.open(run_dir / "mascaras" / "instances" / f"{tile}_instances.png"))
        polys = polygons_of(tile)
        for _, r in g.iterrows():
            m = lab == r.instance_id
            ys, xs = np.nonzero(m)
            pt = Point(xs.mean() + 0.5, ys.mean() + 0.5)
            if not m[min(int(pt.y), m.shape[0] - 1), min(int(pt.x), m.shape[1] - 1)]:
                k = len(xs) // 2  # centroide fora da copa: usa um pixel interno
                pt = Point(xs[k] + 0.5, ys[k] + 0.5)
            hit = [c for poly, c in polys if poly.contains(pt)]
            rows.append({"tile": tile, "instance_id": r.instance_id, "area_m2": r.n_pixels * PIXEL_M ** 2,
                         "seg_prob_mean": r.seg_prob_mean, "pred_class": r.pred_class, "confidence": r.confidence,
                         "e_arvore": bool(hit), "classe_real": hit[0] if hit else 0})
    return pd.DataFrame(rows)


def area_p99_train():
    split = pd.read_csv(P / "unet_spatial_split/tiles_split.csv")
    areas = [poly.area * PIXEL_M ** 2 for t in split[split.split == "train"].tile for poly, _ in polygons_of(t)]
    return float(np.percentile(areas, 99)), len(areas)


def species_table(d, conf):
    rows = []
    for c in sorted(set(d.pred_class) - NO_SPECIES):
        sel = d[(d.pred_class == c) & (d.confidence >= conf) & d.e_arvore]
        rows.append({"classe": int(c), "suporte": len(sel),
                     "precisao": float((sel.classe_real == c).mean()) if len(sel) else float("nan"),
                     "acertos": int((sel.classe_real == c).sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-run", required=True)
    ap.add_argument("--test-run", required=True)
    ap.add_argument("--precisao-arvore", type=float, default=0.95)
    ap.add_argument("--conf-especie", type=float, default=0.70)
    ap.add_argument("--precisao-especie", type=float, default=0.80)
    ap.add_argument("--min-suporte", type=int, default=10)
    ap.add_argument("--saida", default=None)
    args = ap.parse_args()
    out = Path(args.saida) if args.saida else LAB / "runs" / f"osm_limiares_{date.today():%Y%m%d}"
    if (out / "limiares_osm.json").exists():
        raise SystemExit(f"{out} ja tem limiares congelados; nao sobrescrevo.")
    out.mkdir(parents=True, exist_ok=True)

    amax, n_poly = area_p99_train()
    val = label_instances(Path(args.val_run))
    val.to_csv(out / "copas_val.csv", index=False)
    v = val[val.area_m2 <= amax]
    grid = np.round(np.arange(0.50, 0.99, 0.01), 2)
    curve = pd.DataFrame([{"limiar": s, "copas": int((v.seg_prob_mean >= s).sum()),
                           "precisao_arvore": float(v[v.seg_prob_mean >= s].e_arvore.mean()) if (v.seg_prob_mean >= s).any() else float("nan"),
                           "arvores_reais_encontradas": int(v[v.seg_prob_mean >= s].e_arvore.sum())} for s in grid])
    curve.to_csv(out / "validacao_curva_arvore.csv", index=False)
    ok = curve[curve.precisao_arvore >= args.precisao_arvore]
    if ok.empty:
        raise SystemExit("Nenhum limiar atinge a precisao pedida na validacao; reduza --precisao-arvore.")
    s = float(ok.limiar.min())
    vs = v[v.seg_prob_mean >= s]
    sp = species_table(v, args.conf_especie)
    sp["liberada"] = (sp.suporte >= args.min_suporte) & (sp.precisao >= args.precisao_especie)
    sp.to_csv(out / "validacao_especies.csv", index=False)
    limiares = {"area_max_m2": amax, "area_max_origem": f"percentil 99 de {n_poly} poligonos do treino",
                "limiar_seg_prob_mean": s, "precisao_arvore_alvo": args.precisao_arvore,
                "precisao_arvore_val": float(vs.e_arvore.mean()), "copas_val_aprovadas": len(vs),
                "conf_especie": args.conf_especie, "precisao_especie_alvo": args.precisao_especie,
                "min_suporte": args.min_suporte,
                "classes_com_especie": [int(c) for c in sp[sp.liberada].classe],
                "selecionado_em": "validacao", "congelado_em": str(date.today())}
    (out / "limiares_osm.json").write_text(json.dumps(limiares, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- teste: uma unica avaliacao com os limiares congelados ----
    te = label_instances(Path(args.test_run))
    te.to_csv(out / "copas_teste.csv", index=False)
    t_all = te[te.e_arvore]
    ta = te[(te.area_m2 <= amax) & (te.seg_prob_mean >= s)]
    liberadas = limiares["classes_com_especie"]
    spt = species_table(ta, args.conf_especie)                 # no conjunto que iria ao OSM
    spt = spt[spt.classe.isin(liberadas)]
    spa = species_table(te[te.area_m2 <= amax], args.conf_especie)  # em todas as copas validas
    spa = spa[spa.classe.isin(liberadas)]
    res = {"copas_detectadas": len(te), "copas_aprovadas": len(ta),
           "precisao_arvore": float(ta.e_arvore.mean()) if len(ta) else None,
           "precisao_arvore_sem_filtro": float(te.e_arvore.mean()),
           "fracao_das_copas_sobre_arvore_mantidas": float(len(ta[ta.e_arvore]) / max(len(t_all), 1)),
           "especies_no_conjunto_enviado": spt.to_dict(orient="records"),
           "especies_em_todas_as_copas": spa.to_dict(orient="records"),
           "vies_de_especie_seg_prob_mean_val": v.groupby("pred_class").seg_prob_mean.median().round(3).to_dict()}
    (out / "teste_resultado.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"limiares": limiares, "teste": res}, indent=2, ensure_ascii=False))
    print(sp.to_string())


if __name__ == "__main__":
    main()
