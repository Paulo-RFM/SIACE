#!/usr/bin/env python3
"""Repeticao com sementes: SegFormer MiT-B1 x U-Net ResNet-34 (corrigida), sementes 42, 1, 2 e 3 (2026-11-10).

Mesmas sementes da repeticao da U-Net (seeds_corrigido_s{1,2,3}_20261106 + alinhado_20261105_2215_softAug4).
SegFormer: semente 42 = arq_segformer_mit_b1_20261110; sementes 1-3 = seeds_segformer_s<k>_20261110.
Cada run ja congela o proprio limiar na validacao; o teste e avaliado uma vez por run
(evaluate_seg_arquiteturas_test.py) e as metricas vem das mesmas funcoes de avaliar_alinhamento.py.
Comparacao como em avaliar_seeds.py: media, desvio-padrao, diferencas pareadas por semente e teste de
permutacao exato bicaudal (70 divisoes de 8 runs em 4 + 4).

Criterio definido ANTES dos resultados: a SegFormer substitui a U-Net no sistema se a media do IoU de
validacao (criterio de selecao do protocolo) E a media do IoU de teste forem maiores que as da U-Net.
Saida: runs/seeds_segformer_comparacao_20261110/. Runs de treino nao sao alterados.
"""
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from avaliar_alinhamento import IMAGES, MASKS, SPLIT, counts_from_pngs, metrics
from unet_spatial_protocol import load_split, paired_tiles, valid_pixel_mask

LAB = Path(__file__).resolve().parent
R = LAB / "runs"
OUT = R / "seeds_segformer_comparacao_20261110"
SEEDS = [42, 1, 2, 3]
UNET = {42: (R / "alinhado_20261105_2215_softAug4",
             R / "alinhamento_comparacao_20261105" / "softAug4" / "test_eval" / "predictions_test")}
UNET.update({s: (R / f"seeds_corrigido_s{s}_20261106",
                 R / "seeds_alinhamento_comparacao_20261106" / f"seeds_corrigido_s{s}_20261106" / "test_eval"
                 / "predictions_test") for s in (1, 2, 3)})
SEGF = {42: R / "arq_segformer_mit_b1_20261110"}
SEGF.update({s: R / f"seeds_segformer_s{s}_20261110" for s in (1, 2, 3)})
METRICAS = ["iou_val", "iou", "dice_f1", "precision", "recall", "accuracy", "borda_f1_2px", "borda_f1_5px"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = paired_tiles(IMAGES, MASKS)
    split, _, _ = load_split(SPLIT, [i for i, _ in pairs])
    mask_by = {Path(i).stem: m for i, m in pairs}
    img_by = {Path(i).stem: i for i, _ in pairs}
    tiles = sorted(Path(i).stem for i in split["test"])
    gt_by = {t: np.any(np.asarray(Image.open(mask_by[t]).convert("RGB")) != 0, axis=2) for t in tiles}
    valid_by = {t: valid_pixel_mask(Image.open(img_by[t])) for t in tiles}

    runs = {("U-Net", s): v for s, v in UNET.items()}
    for s, run in SEGF.items():
        if not (run / "checkpoint_best.pth").exists() or not (run / "frozen_threshold.json").exists():
            print(f"{run.name}: incompleto, pulando", flush=True)
            continue
        if s == 42:
            ev = R / "arquiteturas_comparacao_20261110" / run.name / "test_eval"
        else:
            ev = OUT / run.name / "test_eval"
        if not (ev / "test_metrics.csv").exists():
            subprocess.run([sys.executable, str(LAB / "evaluate_seg_arquiteturas_test.py"), "--arch", "Segformer",
                            "--encoder", "mit_b1", "--images", IMAGES, "--masks", MASKS, "--split", SPLIT,
                            "--checkpoint", str(run / "checkpoint_best.pth"),
                            "--threshold", str(run / "frozen_threshold.json"), "--output", str(ev)], check=True)
        runs[("SegFormer", s)] = (run, ev / "predictions_test")

    rows = []
    for (g, s), (run, pred) in runs.items():
        lim = json.loads((run / "frozen_threshold.json").read_text())
        m = metrics(counts_from_pngs(pred, tiles, gt_by, valid_by))
        rows.append({"modelo": g, "semente": s, "run": run.name, "limiar": round(float(lim["threshold"]), 2),
                     "iou_val": float(lim["validation_iou"]), **m})
        print(g, s, {k: round(v, 4) for k, v in m.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "resultados_por_run.csv", index=False)

    res = []
    for k in METRICAS:
        a = df[df.modelo == "U-Net"].set_index("semente")[k]
        b = df[df.modelo == "SegFormer"].set_index("semente")[k]
        obs = b.mean() - a.mean()
        vals, n = np.concatenate([a.values, b.values]), len(b)
        diffs = []
        for comb in itertools.combinations(range(len(vals)), n):
            mk = np.zeros(len(vals), bool); mk[list(comb)] = True
            diffs.append(vals[mk].mean() - vals[~mk].mean())
        diffs = np.array(diffs)
        comuns = [s for s in SEEDS if s in a.index and s in b.index]
        par = b.loc[comuns] - a.loc[comuns]
        res.append({"metrica": k, "n_unet": len(a), "n_segformer": len(b), "media_unet": a.mean(),
                    "dp_unet": a.std(ddof=1), "media_segformer": b.mean(), "dp_segformer": b.std(ddof=1),
                    "dif_medias": obs, "p_permutacao_bicaudal": float((np.abs(diffs) >= abs(obs) - 1e-12).mean()),
                    "n_permutacoes": len(diffs),
                    "dif_pareadas_por_semente": "; ".join(f"s{s}:{v:+.4f}" for s, v in par.items()),
                    "sementes_segformer_melhor": f"{int((par > 0).sum())}/{len(par)}"})
    rs = pd.DataFrame(res)
    rs.to_csv(OUT / "resumo.csv", index=False)
    print(rs.to_string(), flush=True)
    r = rs.set_index("metrica")
    completo = len(df[df.modelo == "SegFormer"]) == len(SEEDS)
    troca = completo and r.loc["iou_val", "dif_medias"] > 0 and r.loc["iou", "dif_medias"] > 0
    texto = (f"sementes SegFormer completas: {completo}\n"
             f"dif. media IoU validacao (SegFormer - U-Net): {r.loc['iou_val', 'dif_medias']:+.4f}\n"
             f"dif. media IoU teste (SegFormer - U-Net): {r.loc['iou', 'dif_medias']:+.4f}\n"
             f"criterio (ambas > 0, 4 sementes): {'SegFormer substitui a U-Net' if troca else 'U-Net mantida'}\n")
    (OUT / "decisao.txt").write_text(texto)
    print(texto, flush=True)


if __name__ == "__main__":
    main()
