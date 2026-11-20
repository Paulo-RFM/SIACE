#!/usr/bin/env python3
"""Apaga do servidor de TESTES do OSM os nos enviados a partir de um run de cidade substituido (2026-11-06).

Contexto: os 751 nos enviados em 2026-11-05 (changesets 719860, 719867, 719868 e correcao 720132) vieram do run
runs/pipeline_cidade_segprob_20261105, gerado pela U-Net treinada com o crop desalinhado. A monografia passou a usar
a U-Net corrigida (runs/alinhado_20261105_2215_softAug4), com nova cidade e novos limiares do OSM. Antes do novo
envio, os nos antigos sao apagados; senao, o filtro de duplicatas (--dedup-m) de osm_envio_dev.py descartaria as
novas arvores a menos de 3 m deles.

Le runs/osm_envios_dev_registro.csv (nos criados) e runs/osm_correcoes_dev_registro.csv (nos ja apagados na
correcao das bananeiras), consulta a versao atual de cada no no servidor e apaga todos os ainda visiveis em um
changeset. Padrao = simulacao (so lista). Com --enviar e OSM_DEV_TOKEN, apaga e registra em
runs/osm_remocoes_dev_registro.csv. Os registros antigos nao sao alterados.
Servidor fixo: o de testes (osm_envio_dev.API).
"""
import argparse
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd

import osm_envio_dev as oe

CORRECOES = oe.LAB / "runs" / "osm_correcoes_dev_registro.csv"
REMOCOES = oe.LAB / "runs" / "osm_remocoes_dev_registro.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cidade-run", default="runs/pipeline_cidade_segprob_20261105")
    ap.add_argument("--comentario", default="Remove arvores de teste geradas por modelo substituido (TCC SIACE)")
    ap.add_argument("--enviar", action="store_true")
    args = ap.parse_args()

    led = pd.read_csv(oe.LEDGER)
    alvo = led[led.cidade_run == args.cidade_run].copy()
    if CORRECOES.exists():
        cor = pd.read_csv(CORRECOES)
        alvo = alvo[~alvo.node_id.isin(set(cor[cor.acao == "apagar"].node_id))]
    if REMOCOES.exists():
        alvo = alvo[~alvo.node_id.isin(set(pd.read_csv(REMOCOES).node_id))]
    alvo = alvo.dropna(subset=["node_id"])
    alvo["node_id"] = alvo.node_id.astype(int)
    print(f"{len(alvo)} nos do run {args.cidade_run} a apagar (changesets {sorted(alvo.changeset.unique().tolist())})")
    if alvo.empty or not args.enviar:
        print("simulacao: nada enviado (use --enviar com OSM_DEV_TOKEN)")
        return

    import requests
    token = os.environ.get("OSM_DEV_TOKEN")
    if not token:
        raise SystemExit("Defina OSM_DEV_TOKEN.")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "User-Agent": oe.CREATED_BY})
    print("conta no servidor de testes:", s.get(f"{oe.API}/user/details.json", timeout=60).json()["user"]["display_name"])

    atual, ids = {}, alvo.node_id.tolist()
    for i in range(0, len(ids), 200):  # consulta em blocos; nos ja apagados vem com visible="false"
        r = s.get(f"{oe.API}/nodes", params={"nodes": ",".join(map(str, ids[i:i + 200]))}, timeout=120)
        r.raise_for_status()
        for n in ET.fromstring(r.content).iter("node"):
            if n.get("visible", "true") == "true":
                atual[int(n.get("id"))] = n
    alvo = alvo[alvo.node_id.isin(atual)]
    print(f"{len(alvo)} ainda visiveis no servidor")
    if alvo.empty:
        return

    cs = ET.Element("osm")
    c = ET.SubElement(cs, "changeset")
    for k, v in {"created_by": oe.CREATED_BY, "comment": args.comentario, "bot": "yes"}.items():
        ET.SubElement(c, "tag", k=k, v=v)
    r = s.put(f"{oe.API}/changeset/create", data=ET.tostring(cs), headers={"Content-Type": "text/xml"}, timeout=60)
    r.raise_for_status()
    cid = r.text.strip()
    root = ET.Element("osmChange", version="0.6", generator=oe.CREATED_BY)
    dele = ET.SubElement(root, "delete")
    for nid in alvo.node_id:
        n = atual[nid]
        ET.SubElement(dele, "node", id=n.get("id"), version=n.get("version"), changeset=cid,
                      lat=n.get("lat"), lon=n.get("lon"))
    try:
        r = s.post(f"{oe.API}/changeset/{cid}/upload", data=ET.tostring(root), headers={"Content-Type": "text/xml"},
                   timeout=300)
        r.raise_for_status()
    finally:
        s.put(f"{oe.API}/changeset/{cid}/close", timeout=60)
    now = datetime.now(timezone.utc).isoformat()
    out = alvo[["cidade_run", "tree_id", "node_id"]].assign(changeset=int(cid), removido_em=now)
    out.to_csv(REMOCOES, mode="a", header=not REMOCOES.exists(), index=False)
    print(f"changeset {cid}: {len(out)} nos apagados")


if __name__ == "__main__":
    main()
