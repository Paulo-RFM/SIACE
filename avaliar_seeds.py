#!/usr/bin/env python3
"""Repeticao com sementes do modelo principal (U-Net ResNet-34, softAug4): codigo antigo (crop desalinhado)
x codigo corrigido, para separar o efeito da correcao da variacao entre treinos.

Cada run ja congela o proprio limiar na validacao (frozen_threshold.json, criado pelo script de treino).
Aqui: teste avaliado uma vez por run (evaluate_unet_binary_test.py), metricas de pixel e F1 de borda pelas
mesmas funcoes de avaliar_alinhamento.py, e comparacao entre os grupos:
  media e desvio-padrao por grupo, diferenca das medias, diferencas pareadas por semente e teste de
  permutacao exato (todas as divisoes dos runs em dois grupos do mesmo tamanho), bicaudal.
A semente 42 entra com os runs ja existentes: p1_20261101_140519 (antigo) e alinhado_20261105_2215_softAug4.
Saida: --saida (resultados_por_run.csv, resumo.csv, seeds_iou_borda.png). Runs nao sao alterados.
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from avaliar_alinhamento import IMAGES, MASKS, SPLIT, counts_from_pngs, metrics
from unet_spatial_protocol import load_split, paired_tiles, valid_pixel_mask

LAB = Path(__file__).resolve().parent
R = LAB / "runs"
EXISTENTES = {("antigo", 42): R / "p1_20261101_140519" / "test_eval" / "predictions_test",
              ("corrigido", 42): R / "alinhamento_comparacao_20261105" / "softAug4" / "test_eval" / "predictions_test"}
METRICAS = ["iou", "dice_f1", "precision", "recall", "accuracy", "borda_f1_2px", "borda_f1_5px"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", required=True, help="runs/seeds_<grupo>_s<semente>_<stamp>")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--saida", required=True)
    args = ap.parse_args()
    out = Path(args.saida).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pairs = paired_tiles(IMAGES, MASKS)
    split, _, _ = load_split(SPLIT, [i for i, _ in pairs])
    mask_by = {Path(i).stem: m for i, m in pairs}
    img_by = {Path(i).stem: i for i, _ in pairs}
    tiles = sorted(Path(i).stem for i in split["test"])
    gt_by = {t: np.any(np.asarray(Image.open(mask_by[t]).convert("RGB")) != 0, axis=2) for t in tiles}
    valid_by = {t: valid_pixel_mask(Image.open(img_by[t])) for t in tiles}

    preds = dict(EXISTENTES)
    for s in [int(x) for x in args.seeds.split(",")]:
        for g in ("antigo", "corrigido"):
            run = R / f"seeds_{g}_s{s}_{args.stamp}"
            if not (run / "checkpoint_best.pth").exists() or not (run / "frozen_threshold.json").exists():
                print(f"{run.name}: incompleto, pulando", flush=True)
                continue
            ev = out / run.name / "test_eval"
            if not (ev / "test_metrics.csv").exists():
                subprocess.run([sys.executable, str(LAB / "evaluate_unet_binary_test.py"), "--images", IMAGES,
                                "--masks", MASKS, "--split", SPLIT, "--checkpoint", str(run / "checkpoint_best.pth"),
                                "--threshold", str(run / "frozen_threshold.json"), "--output", str(ev)], check=True)
            preds[(g, s)] = ev / "predictions_test"
    rows = []
    for (g, s), p in sorted(preds.items()):
        m = metrics(counts_from_pngs(p, tiles, gt_by, valid_by))
        rows.append({"grupo": g, "semente": s, "predicoes": str(p.relative_to(LAB)), **m})
        print(g, s, {k: round(v, 4) for k, v in m.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "resultados_por_run.csv", index=False)

    res = []
    for k in METRICAS:
        a = df[df.grupo == "antigo"].set_index("semente")[k]
        b = df[df.grupo == "corrigido"].set_index("semente")[k]
        obs = b.mean() - a.mean()
        vals, n = np.concatenate([a.values, b.values]), len(b)
        diffs = []
        for comb in itertools.combinations(range(len(vals)), n):
            mk = np.zeros(len(vals), bool); mk[list(comb)] = True
            diffs.append(vals[mk].mean() - vals[~mk].mean())
        diffs = np.array(diffs)
        comuns = sorted(set(a.index) & set(b.index))
        par = (b.loc[comuns] - a.loc[comuns])
        res.append({"metrica": k, "n_antigo": len(a), "n_corrigido": len(b),
                    "media_antigo": a.mean(), "dp_antigo": a.std(ddof=1), "media_corrigido": b.mean(),
                    "dp_corrigido": b.std(ddof=1), "dif_medias": obs, "dif_rel_pct": 100 * obs / a.mean(),
                    "p_permutacao_bicaudal": float((np.abs(diffs) >= abs(obs) - 1e-12).mean()),
                    "n_permutacoes": len(diffs), "dif_pareadas_por_semente": "; ".join(f"s{s}:{v:+.4f}" for s, v in par.items()),
                    "sementes_com_melhora": f"{int((par > 0).sum())}/{len(par)}"})
    rs = pd.DataFrame(res)
    rs.to_csv(out / "resumo.csv", index=False)
    print(rs.to_string(), flush=True)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, k, ttl in [(axs[0], "iou", "IoU de teste"), (axs[1], "borda_f1_2px", "F1 de borda (2 px)")]:
        for x, g, c in [(0, "antigo", "#9aa5b1"), (1, "corrigido", "#2e6f50")]:
            v = df[df.grupo == g][k]
            ax.scatter(np.full(len(v), x) + np.linspace(-0.08, 0.08, len(v)), v, color=c, s=40, zorder=3)
            ax.hlines(v.mean(), x - 0.2, x + 0.2, color=c, lw=2)
        ax.set_xticks([0, 1], ["com desalinhamento", "corrigido"]); ax.set_title(ttl); ax.grid(axis="y", alpha=.3)
    fig.suptitle("U-Net ResNet-34: um ponto por semente; traco = media")
    fig.tight_layout(); fig.savefig(out / "seeds_iou_borda.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
