#!/usr/bin/env python3
"""Corrige no servidor de TESTES do OSM os nos de bananeira enviados antes como natural=tree (2026-11-05).

A bananeira nao e arvore (planta herbacea perene de grande porte), e osm_envio_dev.py passou a envia-la como
natural=plant + genus=Musa, e so quando a especie e indicada (CLASSES_NAO_ARVORE). Para deixar o servidor coerente
com essa regra, este script le o registro runs/osm_envios_dev_registro.csv, cruza com arvores.csv e:
  * no com especie indicada (com_especie=True) -> modifica as tags para natural=plant + tags da especie;
  * no sem especie (confianca abaixo do limiar)  -> apaga o no (nao ha certeza de que seja arvore nem bananeira).
Tudo em um changeset. Padrao = simulacao (so lista); com --enviar e OSM_DEV_TOKEN, aplica e registra em
runs/osm_correcoes_dev_registro.csv (nos ja corrigidos sao pulados).
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import osm_envio_dev as oe

CORRECOES = oe.LAB / "runs" / "osm_correcoes_dev_registro.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cidade-run", required=True)
    ap.add_argument("--tags", default=str(oe.LAB / "osm_especies_tags.json"))
    ap.add_argument("--comentario", default="Bananeiras: natural=tree -> natural=plant (planta herbacea; teste TCC SIACE)")
    ap.add_argument("--enviar", action="store_true")
    args = ap.parse_args()
    tags_cfg = json.loads(Path(args.tags).read_text())
    if not tags_cfg.get("confirmado_pelo_autor"):
        raise SystemExit("osm_especies_tags.json nao confirmado pelo autor.")

    led = pd.read_csv(oe.LEDGER)
    led = led[led.cidade_run == args.cidade_run]
    trees = pd.read_csv(Path(args.cidade_run) / "arvores.csv")[["tree_id", "pred_class", "pred_name"]]
    alvo = led.merge(trees, on="tree_id")
    alvo = alvo[alvo.pred_class.isin(set(oe.CLASSES_NAO_ARVORE))].copy()
    if CORRECOES.exists():
        alvo = alvo[~alvo.node_id.isin(set(pd.read_csv(CORRECOES).node_id))]
    alvo["acao"] = alvo.com_especie.map({True: "modificar", False: "apagar"})
    print(f"{len(alvo)} nos:", alvo[["tree_id", "node_id", "pred_name", "com_especie", "acao"]].to_string(index=False),
          sep="\n")
    if alvo.empty or not args.enviar:
        return

    import requests
    token = os.environ.get("OSM_DEV_TOKEN")
    if not token:
        raise SystemExit("Defina OSM_DEV_TOKEN.")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "User-Agent": oe.CREATED_BY})
    atual = {}
    for nid in alvo.node_id.astype(int):
        r = s.get(f"{oe.API}/node/{nid}", timeout=60)
        r.raise_for_status()
        atual[nid] = ET.fromstring(r.content).find("node")

    cs = ET.Element("osm")
    c = ET.SubElement(cs, "changeset")
    for k, v in {"created_by": oe.CREATED_BY, "comment": args.comentario, "bot": "yes"}.items():
        ET.SubElement(c, "tag", k=k, v=v)
    r = s.put(f"{oe.API}/changeset/create", data=ET.tostring(cs), headers={"Content-Type": "text/xml"}, timeout=60)
    r.raise_for_status()
    cid = r.text.strip()
    root = ET.Element("osmChange", version="0.6", generator=oe.CREATED_BY)
    mod, dele = ET.SubElement(root, "modify"), ET.SubElement(root, "delete")
    for row in alvo.itertuples():
        n = atual[int(row.node_id)]
        attrs = dict(id=n.get("id"), version=n.get("version"), changeset=cid, lat=n.get("lat"), lon=n.get("lon"))
        if row.com_especie:
            m = ET.SubElement(mod, "node", **attrs)
            for k, v in oe.node_tags(row, tags_cfg).items():
                ET.SubElement(m, "tag", k=k, v=str(v))
        else:
            ET.SubElement(dele, "node", **attrs)
    try:
        r = s.post(f"{oe.API}/changeset/{cid}/upload", data=ET.tostring(root), headers={"Content-Type": "text/xml"},
                   timeout=120)
        r.raise_for_status()
    finally:
        s.put(f"{oe.API}/changeset/{cid}/close", timeout=60)
    now = datetime.now(timezone.utc).isoformat()
    out = alvo[["tree_id", "node_id", "pred_name", "acao"]].assign(cidade_run=args.cidade_run, changeset=int(cid),
                                                                     corrigido_em=now)
    out.to_csv(CORRECOES, mode="a", header=not CORRECOES.exists(), index=False)
    print(f"changeset {cid}: {int((alvo.acao == 'modificar').sum())} modificados, "
          f"{int((alvo.acao == 'apagar').sum())} apagados")


if __name__ == "__main__":
    main()
