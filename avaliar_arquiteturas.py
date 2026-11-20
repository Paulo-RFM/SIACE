#!/usr/bin/env python3
"""Benchmark de arquiteturas de segmentacao binaria (2026-11-10).

Compara a U-Net ResNet-34 principal (alinhado_20261105_2215_softAug4) com U-Net++, DeepLabV3+ e SegFormer,
treinadas por tree_segmentation_arquiteturas.py com o mesmo protocolo. Para cada arquitetura nova:
  1. avalia o teste UMA vez com o limiar congelado na validacao (evaluate_seg_arquiteturas_test.py);
  2. calcula as metricas de pixel e de borda com as funcoes de avaliar_alinhamento.py;
  3. calcula IC 95% por bootstrap por tile e a diferenca pareada de IoU contra a U-Net principal,
     com as funcoes, B = 2000 e semente 42 de bootstrap_ic_corrigido.py.
A U-Net principal entra com as predicoes de teste ja salvas (nada dela e reavaliado).
Saida: runs/arquiteturas_comparacao_20261110/{resultados.csv, ic_95.csv}. Runs de treino nao sao alterados.
"""
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from avaliar_alinhamento import IMAGES, MASKS, SPLIT, counts_from_pngs, metrics
from bootstrap_ic_corrigido import SEED, boot_idx, ci, iou_f1, seg_counts
from unet_spatial_protocol import load_split, paired_tiles, valid_pixel_mask

LAB = Path(__file__).resolve().parent
R = LAB / "runs"
OUT = R / "arquiteturas_comparacao_20261110"
PRINCIPAL = "U-Net (ResNet-34)"
MODELOS = {  # nome -> (run de treino, arquitetura, codificador, pasta das predicoes de teste ou None)
    PRINCIPAL: ("alinhado_20261105_2215_softAug4", "Unet", "resnet34",
                R / "alinhamento_comparacao_20261105" / "softAug4" / "test_eval"),
    "U-Net++ (ResNet-34)": ("arq_unetplusplus_resnet34_20261110", "UnetPlusPlus", "resnet34", None),
    "DeepLabV3+ (ResNet-34)": ("arq_deeplabv3plus_resnet34_20261110", "DeepLabV3Plus", "resnet34", None),
    "SegFormer (MiT-B1)": ("arq_segformer_mit_b1_20261110", "Segformer", "mit_b1", None),
}


def minutos_treino(run_dir):
    texto = (run_dir / "log.txt").read_text(errors="ignore") if (run_dir / "log.txt").exists() else ""
    inicio = re.search(r"started_at=(\S+)", texto)
    fim = re.search(r"ended_at=(\S+?)(?:\s|$)", texto)
    if not (inicio and fim):
        return float("nan")
    return (datetime.fromisoformat(fim.group(1)) - datetime.fromisoformat(inicio.group(1))).total_seconds() / 60


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pares = paired_tiles(IMAGES, MASKS)
    split, _, _ = load_split(SPLIT, [i for i, _ in pares])
    img_by = {Path(i).stem: i for i, _ in pares}
    mask_by = {Path(i).stem: m for i, m in pares}
    tiles = [Path(i).stem for i in split["test"]]  # mesma ordem de bootstrap_ic_corrigido.py
    gt_by = {t: np.any(np.asarray(Image.open(mask_by[t]).convert("RGB")) != 0, axis=2) for t in tiles}
    valid_by = {t: valid_pixel_mask(Image.open(img_by[t])) for t in tiles}

    avaliacoes = {}
    for nome, (run, arch, enc, ev) in MODELOS.items():
        run_dir = R / run
        if ev is None:
            if not (run_dir / "checkpoint_best.pth").exists() or not (run_dir / "frozen_threshold.json").exists():
                print(f"{nome}: run {run} incompleto, pulando", flush=True)
                continue
            ev = OUT / run / "test_eval"
            if not (ev / "test_metrics.csv").exists():
                subprocess.run([sys.executable, str(LAB / "evaluate_seg_arquiteturas_test.py"),
                                "--arch", arch, "--encoder", enc, "--images", IMAGES, "--masks", MASKS,
                                "--split", SPLIT, "--checkpoint", str(run_dir / "checkpoint_best.pth"),
                                "--threshold", str(run_dir / "frozen_threshold.json"), "--output", str(ev)],
                               check=True)
        avaliacoes[nome] = (run_dir, ev)

    rng = np.random.default_rng(SEED)
    idx = boot_idx(len(tiles), rng)
    base = seg_counts(avaliacoes[PRINCIPAL][1] / "predictions_test", tiles, gt_by, valid_by)
    linhas, ic = [], []
    for nome, (run_dir, ev) in avaliacoes.items():
        pred = ev / "predictions_test"
        m = metrics(counts_from_pngs(pred, tiles, gt_by, valid_by))
        c = seg_counts(pred, tiles, gt_by, valid_by)
        amostras = np.array([iou_f1(c[i]) for i in idx])
        iou_lo, iou_hi = ci(amostras[:, 0])
        f1_lo, f1_hi = ci(amostras[:, 1])
        limiar = json.loads((run_dir / "frozen_threshold.json").read_text())
        hist = pd.read_csv(run_dir / "metrics.csv")
        teste = pd.read_csv(ev / "test_metrics.csv").iloc[0]
        linha = {
            "modelo": nome, "run": run_dir.name, "parametros": int(hist["trainable_parameters"].iloc[0])
            if "trainable_parameters" in hist else "",
            "epocas": len(hist), "melhor_epoca": int(hist.loc[hist["iou"].idxmax(), "epoch"]),
            "minutos_treino": round(minutos_treino(run_dir), 1),
            "pico_gpu_mib": round(float(hist["gpu_mem_alloc_mb"].max()), 0) if "gpu_mem_alloc_mb" in hist else "",
            "limiar": round(float(limiar["threshold"]), 2), "iou_val": limiar["validation_iou"],
            **m, "iou_ic95_inf": iou_lo, "iou_ic95_sup": iou_hi, "f1_ic95_inf": f1_lo, "f1_ic95_sup": f1_hi,
            "inferencia_s_por_tile": float(teste["inference_per_tile_s"]),
        }
        if nome != PRINCIPAL:
            dif = [iou_f1(c[i])[0] - iou_f1(base[i])[0] for i in idx]
            lo, hi = ci(dif)
            linha.update({"dif_iou_vs_unet": iou_f1(c)[0] - iou_f1(base)[0], "dif_ic95_inf": lo, "dif_ic95_sup": hi})
            ic.append({"comparacao": f"{nome} - {PRINCIPAL}", "metrica": "dif. IoU teste (pareada)",
                       "estimativa": linha["dif_iou_vs_unet"], "ic95_inf": lo, "ic95_sup": hi, "n_tiles": len(tiles)})
        ic.append({"comparacao": nome, "metrica": "IoU teste", "estimativa": m["iou"], "ic95_inf": iou_lo,
                   "ic95_sup": iou_hi, "n_tiles": len(tiles)})
        linhas.append(linha)
        print(nome, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in linha.items()}, flush=True)

    pd.DataFrame(linhas).to_csv(OUT / "resultados.csv", index=False)
    pd.DataFrame(ic).to_csv(OUT / "ic_95.csv", index=False)
    print("Saida:", OUT, flush=True)


if __name__ == "__main__":
    main()
