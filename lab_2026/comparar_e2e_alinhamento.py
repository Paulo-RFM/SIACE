#!/usr/bin/env python3
"""Deteccao de ponta a ponta no teste: U-Net principal antiga (crop desalinhado) x corrigida.

Mesmo pipeline e mesma avaliacao do baseline (pipeline_cidade.py --partition test com componentes
conexas min_area 3000 e a mesma EfficientNet; evaluate_pipeline_test.py com IoU >= 0,5). So muda o
checkpoint de segmentacao. Compara, com bootstrap pareado por tile (B=2000, seed 42):
  F1, precisao e recall de deteccao; instancias por arvore anotada (< 1 indica copas fundidas);
  erro absoluto medio da contagem por tile; e, por tile, arvores anotadas cobertas por uma mesma
  instancia predita (fusao de copas vizinhas).
Gera tambem exemplos visuais (contornos antigo, corrigido e anotado) nos tiles de maior mudanca de IoU.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

LAB = Path(__file__).resolve().parent
R = LAB / "runs"
PY = sys.executable
CLS = R / "classif_20261102_163825_efficientnet_anotados"
OLD_PIPE = R / "pipeline_test_cc_20261103_223209"
IMAGES, MASKS = Path("/home/paulomaciel/proj/imagens"), Path("/home/paulomaciel/proj/masks")
B, SEED = 2000, 42


def det(d):
    tp, n_gt, n_pred = d.tp.sum(), d.n_gt.sum(), d.n_pred.sum()
    p, r = tp / max(n_pred, 1), tp / max(n_gt, 1)
    return {"f1": 2 * p * r / max(p + r, 1e-12), "precisao": p, "recall": r, "instancias_por_arvore": n_pred / n_gt,
            "mae_contagem": float(np.abs(d.n_pred - d.n_gt).mean()), "fusoes": int(d.fusoes.sum())}


def fusions(pipe, tiles):
    """Arvores anotadas a mais em instancias que cobrem o centro de 2+ arvores anotadas."""
    sys.path.insert(0, str(LAB))
    from evaluate_pipeline_test import gt_polygons
    out = {}
    for t in tiles:
        p = pipe / "mascaras" / "instances" / f"{t}_instances.png"
        inst = np.array(Image.open(p)).astype(np.int32) if p.is_file() else np.zeros((512, 512), np.int32)
        hits = {}
        for m, _ in gt_polygons(t):
            cy, cx = ndi.center_of_mass(m)
            i = inst[int(round(cy)), int(round(cx))]
            if i > 0:
                hits[i] = hits.get(i, 0) + 1
        out[t] = sum(v - 1 for v in hits.values() if v > 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg-run", required=True, help="run corrigido da configuracao principal")
    ap.add_argument("--pred-novo", required=True, help="test_eval/predictions_test do run corrigido")
    ap.add_argument("--pipeline-dir", required=True)
    ap.add_argument("--saida", required=True)
    args = ap.parse_args()
    pipe, out = Path(args.pipeline_dir), Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    if not (pipe / "instancias.csv").exists():
        subprocess.run([PY, str(LAB / "pipeline_cidade.py"), "--tiles-dir", str(IMAGES), "--partition", "test",
                        "--seg-run", args.seg_run, "--cls-run", str(CLS), "--output-dir", str(pipe), "--save-masks"],
                       check=True)
    if not (pipe / "avaliacao_teste" / "metricas_teste.json").exists():
        subprocess.run([PY, str(LAB / "evaluate_pipeline_test.py"), "--pipeline-dir", str(pipe),
                        "--p1-predictions", args.pred_novo], check=True)
    a = pd.read_csv(OLD_PIPE / "avaliacao_teste" / "deteccao_por_tile.csv").set_index("tile")
    b = pd.read_csv(pipe / "avaliacao_teste" / "deteccao_por_tile.csv").set_index("tile")
    tiles = sorted(set(a.index) & set(b.index))
    a, b = a.loc[tiles].copy(), b.loc[tiles].copy()
    fa, fb = fusions(OLD_PIPE, tiles), fusions(pipe, tiles)
    a["fusoes"] = [fa[t] for t in tiles]; b["fusoes"] = [fb[t] for t in tiles]
    ma, mb = det(a), det(b)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(B):
        i = rng.integers(0, len(tiles), len(tiles))
        boots.append((det(a.iloc[i]), det(b.iloc[i])))
    rows = []
    for k in ma:
        so = np.array([x[0][k] for x in boots], float); sn = np.array([x[1][k] for x in boots], float); d = sn - so
        rows.append({"metrica": k, "antigo": ma[k], "novo": mb[k], "dif_abs": mb[k] - ma[k],
                     "dif_rel_pct": 100 * (mb[k] - ma[k]) / ma[k] if ma[k] else float("nan"),
                     "dif_ic95_inf": float(np.percentile(d, 2.5)), "dif_ic95_sup": float(np.percentile(d, 97.5))})
    for nm, p in [("antigo", OLD_PIPE), ("novo", pipe)]:
        js = json.loads((p / "avaliacao_teste" / "metricas_teste.json").read_text())
        rows.append({"metrica": f"classificacao_acuracia_{nm}", nm: js["classificacao_nas_arvores_detectadas"]["accuracy"]})
        rows.append({"metrica": f"classificacao_f1_macro_{nm}", nm: js["classificacao_nas_arvores_detectadas"]["f1_macro"]})
        rows.append({"metrica": f"n_tp_{nm}", nm: js["deteccao"]["tp"]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "e2e_comparacao.csv", index=False)
    print(df.to_string())
    # exemplos visuais: tiles de maior mudanca de IoU de pixel (contagens_por_tile.csv da comparacao)
    ct = out / "contagens_por_tile.csv"
    if ct.exists():
        c = pd.read_csv(ct)
        c = c[c.config == "softAug4"].copy()
        iou = lambda p: c[f"{p}_tp"] / (c[f"{p}_tp"] + c[f"{p}_fp"] + c[f"{p}_fn"]).clip(lower=1)
        c["d_iou"] = iou("novo") - iou("antigo")
        sel = list(c.nlargest(4, "d_iou").tile) + list(c.nsmallest(2, "d_iou").tile)
        old_pred = R / "p1_20261101_140519" / "test_eval" / "predictions_test"
        for k, t in enumerate(sel):
            img = np.array(Image.open(IMAGES / f"{t}.tif").convert("RGB")) if (IMAGES / f"{t}.tif").exists() else \
                np.array(Image.open(IMAGES / f"{t}.png").convert("RGB"))
            gt = np.any(np.array(Image.open(MASKS / f"{t}.png").convert("RGB")) != 0, axis=2)
            fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
            for ax, m, ttl in [(axs[0], gt, "anotado"),
                               (axs[1], np.array(Image.open(old_pred / f"{t}.png")) > 127, "com desalinhamento"),
                               (axs[2], np.array(Image.open(Path(args.pred_novo) / f"{t}.png")) > 127, "corrigido")]:
                edge = ndi.binary_dilation(m, iterations=2) & ~ndi.binary_erosion(m, iterations=2)
                ov = np.zeros(m.shape + (4,)); ov[edge] = (1, 1, 0, 1)
                ax.imshow(img); ax.imshow(ov); ax.set_title(ttl); ax.axis("off")
            row = c[c.tile == t].iloc[0]
            fig.suptitle(f"{t}: IoU {iou('antigo')[row.name]:.3f} -> {iou('novo')[row.name]:.3f}")
            fig.tight_layout(); fig.savefig(out / f"exemplo_teste_{k:02d}_{t}.png", dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
