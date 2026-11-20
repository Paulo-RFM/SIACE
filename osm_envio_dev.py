#!/usr/bin/env python3
"""Envio em massa de arvores detectadas para o servidor de TESTES do OpenStreetMap (API 0.6).

Servidor fixo: https://master.apis.dev.openstreetmap.org (dados de teste, separados do OSM real).
Nao ha opcao para o servidor de producao: um envio automatico ao OSM real exige o processo de importacao
da comunidade (proposta, discussao, conta dedicada), fora do escopo deste script.

Bananeiras (classe 5) nao sao arvores: vao como natural=plant + genus=Musa, e so quando a especie e indicada.
Selecao (limiares congelados na validacao por osm_calibrar_limiares.py -> limiares_osm.json):
  * natural=tree so para copas com area <= area_max_m2 e seg_prob_mean >= limiar_seg_prob_mean;
  * especie so se a classe estiver em classes_com_especie, a confianca do classificador for >= conf_especie
    e osm_especies_tags.json estiver confirmado pelo autor; senao o no vai apenas com natural=tree.
Evita duplicatas: (1) registro local dos envios (runs/osm_envios_dev_registro.csv) e (2) com --enviar,
consulta as arvores ja existentes no servidor de testes e descarta candidatas a menos de --dedup-m metros.

Padrao = simulacao: gera os arquivos .osc, selecao.geojson e resumo.json em --saida, sem enviar nada.
Para enviar: crie uma conta em https://master.apis.dev.openstreetmap.org, registre um aplicativo OAuth 2
com os escopos read_prefs e write_api, obtenha um token e exporte OSM_DEV_TOKEN=<token>; depois rode com --enviar.
Comece com --limite 5 e confira o resultado no site antes de mandar o restante.

Exemplo:
  python osm_envio_dev.py --cidade-run runs/<cidade> --limiares runs/osm_limiares_<data>/limiares_osm.json
  OSM_DEV_TOKEN=... python osm_envio_dev.py ... --enviar --limite 5
"""
import argparse
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LAB = Path(__file__).resolve().parent
API = "https://master.apis.dev.openstreetmap.org/api/0.6"
LEDGER = LAB / "runs" / "osm_envios_dev_registro.csv"
CREATED_BY = "SIACE osm_envio_dev.py"
# Classes do inventario que NAO sao arvores: vao ao OSM com natural=plant (etiqueta generica de planta, "in use"
# no wiki do OSM), e nao natural=tree. Bananeira: planta herbacea perene de grande porte (pseudocaule de bainhas
# foliares), nao lenhosa. Como "planta" so faz sentido se a especie for indicada, essas copas so sao enviadas
# quando a especie e indicada (com_especie); senao ficam de fora. Decisao do autor, 2026-11-05.
# Os nos de bananeira enviados antes como natural=tree sao corrigidos com osm_corrigir_bananeiras_dev.py.
CLASSES_NAO_ARVORE = {5: "plant"}


def haversine_m(lon1, lat1, lon2, lat2):
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def select(trees, lim, tags_cfg):
    if "seg_prob_mean" not in trees.columns or trees["seg_prob_mean"].isna().any():
        raise SystemExit("arvores.csv sem seg_prob_mean: use um run do pipeline atualizado (2026-11-04+).")
    sel = trees[(trees.area_m2 <= lim["area_max_m2"]) & (trees.seg_prob_mean >= lim["limiar_seg_prob_mean"])].copy()
    sel["especie_elegivel"] = sel.pred_class.isin(set(lim["classes_com_especie"])) & (sel.confidence >= lim["conf_especie"])
    sel["com_especie"] = sel.especie_elegivel & bool(tags_cfg.get("confirmado_pelo_autor"))
    return sel[~sel.pred_class.isin(set(CLASSES_NAO_ARVORE)) | sel.com_especie]


def node_tags(row, tags_cfg):
    tags = {"natural": CLASSES_NAO_ARVORE.get(int(row.pred_class), "tree") if row.com_especie else "tree"}
    if row.com_especie:
        tags.update(tags_cfg["classes"][str(int(row.pred_class))]["tags"])
    return tags


def osc_xml(rows, tags_cfg, changeset="0"):
    root = ET.Element("osmChange", version="0.6", generator=CREATED_BY)
    create = ET.SubElement(root, "create")
    for k, (_, r) in enumerate(rows.iterrows(), start=1):
        n = ET.SubElement(create, "node", id=str(-k), lat=f"{r.lat:.7f}", lon=f"{r.lon:.7f}", changeset=str(changeset))
        for key, val in node_tags(r, tags_cfg).items():
            ET.SubElement(n, "tag", k=key, v=str(val))
    return ET.tostring(root, encoding="unicode")


def existing_trees(session, sel, step=0.005):
    """Arvores (natural=tree) ja no servidor de testes, consultadas por blocos de bbox."""
    pts = []
    lon0, lon1, lat0, lat1 = sel.lon.min(), sel.lon.max(), sel.lat.min(), sel.lat.max()
    lo = lon0
    while lo <= lon1:
        la = lat0
        while la <= lat1:
            bbox = f"{lo:.6f},{la:.6f},{min(lo + step, lon1 + 1e-6):.6f},{min(la + step, lat1 + 1e-6):.6f}"
            r = session.get(f"{API}/map", params={"bbox": bbox}, timeout=120)
            r.raise_for_status()
            for n in ET.fromstring(r.content).iter("node"):
                if any(t.get("k") == "natural" and t.get("v") == "tree" for t in n.iter("tag")):
                    pts.append((float(n.get("lon")), float(n.get("lat"))))
            la += step
            time.sleep(0.5)
        lo += step
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cidade-run", required=True)
    ap.add_argument("--limiares", required=True)
    ap.add_argument("--tags", default=str(LAB / "osm_especies_tags.json"))
    ap.add_argument("--saida", default=None)
    ap.add_argument("--lote", type=int, default=500, help="nos por changeset (API aceita ate 10.000 mudancas)")
    ap.add_argument("--limite", type=int, default=None, help="envia apenas as N primeiras arvores selecionadas")
    ap.add_argument("--dedup-m", type=float, default=3.0)
    ap.add_argument("--comentario", default="Teste: arvores urbanas detectadas por VANT (Rio Paranaiba-MG), TCC SIACE")
    ap.add_argument("--enviar", action="store_true", help="envia de fato ao servidor de testes")
    args = ap.parse_args()

    city = Path(args.cidade_run)
    lim = json.loads(Path(args.limiares).read_text())
    tags_cfg = json.loads(Path(args.tags).read_text())
    trees = pd.read_csv(city / "arvores.csv")
    sel = select(trees, lim, tags_cfg)
    if LEDGER.exists():
        led = pd.read_csv(LEDGER)
        done = set(led[led.cidade_run == str(city)].tree_id)
        sel = sel[~sel.tree_id.isin(done)]
    sel = sel.sort_values("tree_id")
    if args.limite:
        sel = sel.head(args.limite)
    out = Path(args.saida) if args.saida else LAB / "runs" / f"osm_envio_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=False)

    session = None
    if args.enviar:
        import requests
        token = os.environ.get("OSM_DEV_TOKEN")
        if not token:
            raise SystemExit("Defina OSM_DEV_TOKEN (token OAuth 2 do servidor de testes, escopos read_prefs e write_api).")
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}", "User-Agent": CREATED_BY})
        me = session.get(f"{API}/user/details.json", timeout=60)
        me.raise_for_status()
        print("conta no servidor de testes:", me.json()["user"]["display_name"], flush=True)
        if args.dedup_m > 0 and len(sel):
            ex = existing_trees(session, sel)
            near = [any(haversine_m(r.lon, r.lat, x, y) < args.dedup_m for x, y in ex) for r in sel.itertuples()]
            print(f"{sum(near)} candidatas descartadas por ja existir arvore a < {args.dedup_m} m", flush=True)
            sel = sel[[not n for n in near]]

    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
              "properties": {"tree_id": int(r.tree_id), "pred_name": r.pred_name, "confidence": float(r.confidence),
                             "seg_prob_mean": float(r.seg_prob_mean), "area_m2": float(r.area_m2),
                             "com_especie": bool(r.com_especie),
                             **{f"osm:{k}": v for k, v in node_tags(r, tags_cfg).items()}}}
             for r in sel.itertuples()]
    (out / "selecao.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    batches = [sel.iloc[i:i + args.lote] for i in range(0, len(sel), args.lote)]
    for b, rows in enumerate(batches):
        (out / f"lote_{b:03d}.osc").write_text(osc_xml(rows, tags_cfg), encoding="utf-8")
    resumo = {"cidade_run": str(city), "limiares": lim, "tags_confirmadas": bool(tags_cfg.get("confirmado_pelo_autor")),
              "arvores_no_run": len(trees), "selecionadas": len(sel), "com_especie": int(sel.com_especie.sum()),
              "especie_elegivel": int(sel.especie_elegivel.sum()),
              "elegiveis_por_especie": sel[sel.especie_elegivel].pred_name.value_counts().to_dict(),
              "por_especie": sel[sel.com_especie].pred_name.value_counts().to_dict(),
              "lotes": len(batches), "enviado": False, "servidor": API}
    print(json.dumps({k: v for k, v in resumo.items() if k != "limiares"}, indent=2, ensure_ascii=False), flush=True)

    if args.enviar:
        log = []
        for b, rows in enumerate(batches):
            cs = ET.Element("osm")
            c = ET.SubElement(cs, "changeset")
            for k, v in {"created_by": CREATED_BY, "comment": args.comentario, "bot": "yes",
                         "source": "ortofoto de VANT; deteccao automatica (SegFormer MiT-B1 + EfficientNet-B0)"}.items():
                ET.SubElement(c, "tag", k=k, v=v)
            r = session.put(f"{API}/changeset/create", data=ET.tostring(cs), headers={"Content-Type": "text/xml"},
                            timeout=60)
            r.raise_for_status()
            cid = r.text.strip()
            try:
                r = session.post(f"{API}/changeset/{cid}/upload", data=osc_xml(rows, tags_cfg, cid).encode(),
                                 headers={"Content-Type": "text/xml"}, timeout=300)
                r.raise_for_status()
                ids = {int(n.get("old_id")): int(n.get("new_id")) for n in ET.fromstring(r.content).iter("node")}
            finally:
                session.put(f"{API}/changeset/{cid}/close", timeout=60)
            now = datetime.now(timezone.utc).isoformat()
            for k, row in enumerate(rows.itertuples(), start=1):
                log.append({"cidade_run": str(city), "tree_id": int(row.tree_id), "node_id": ids.get(-k),
                            "changeset": int(cid), "com_especie": bool(row.com_especie), "enviado_em": now})
            print(f"lote {b}: changeset {cid}, {len(ids)} nos criados", flush=True)
            pd.DataFrame(log).to_csv(LEDGER, mode="a", header=not LEDGER.exists(), index=False)
            log = []
            time.sleep(1)
        resumo["enviado"] = True
    (out / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
