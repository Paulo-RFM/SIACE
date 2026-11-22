#!/usr/bin/env python3
"""Reavalia as configuracoes herdadas do benchmark de segmentacao com limiar escolhido NA VALIDACAO.

Mesmo protocolo da configuracao principal (tree_segmentation_resnet_26_512softAug4.py):
  1. predicao do checkpoint_best.pth nos 203 tiles de validacao;
  2. varredura de limiar 0,10..0,90 (passo 0,05) e escolha do maior IoU de validacao
     (freeze_validation_threshold, mesmo codigo do run principal);
  3. limiar congelado em arquivo ANTES do teste;
  4. avaliacao unica no teste com evaluate_unet_binary_test.py (mesmo script do run principal).

Nao altera nada dentro de runs/<run>/: tudo e gravado em runs/limiar_val_herdados_20261104/<run>/.
O run principal entra apenas na etapa de validacao, como verificacao de que esta reimplementacao
reproduz o limiar 0,65 ja congelado (o teste dele nao e refeito).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from segmentation_models_pytorch import Unet
from torch.utils.data import DataLoader

from evaluate_unet_binary_test import TestDataset
from unet_spatial_protocol import freeze_validation_threshold, load_split, paired_tiles

LAB = Path(__file__).resolve().parent
IMAGES = "/home/paulomaciel/proj/imagens"
MASKS = "/home/paulomaciel/proj/masks"
SPLIT = "/home/paulomaciel/proj/unet_spatial_split"
OUT = LAB / "runs" / "limiar_val_herdados_20261104"
RUNS = ["20261101_185936_base", "20261101_185936_26", "20261101_185936_512semAug",
        "treino_20261101_203446_512semAug2",
        "treino_20261101_203446_512softAug2"]
# 20261101_185936_512semAug2 ficou de fora: treino interrompido (log termina em "Sem melhora 5/7"); o mesmo script foi refeito em treino_20261101_203446_512semAug2.
CHECK_RUN = "p1_20261101_140519"


def val_probabilities(checkpoint, images, masks, device):
    ds = TestDataset(images, masks, (512, 512))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    model = Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    probs, targets, valids = [], [], []
    with torch.no_grad():
        for image, target, valid in loader:
            probs.append(torch.sigmoid(model(image.to(device)))[0, 0].cpu().numpy().astype(np.float32))
            targets.append(target[0, 0].numpy().astype(np.uint8))
            valids.append(valid[0].numpy())
    del model
    torch.cuda.empty_cache()
    return np.stack(probs), np.stack(targets), np.stack(valids)


def main():
    pairs = paired_tiles(IMAGES, MASKS)
    split, _, _ = load_split(SPLIT, [i for i, _ in pairs])
    mask_by = {Path(i).stem: m for i, m in pairs}
    val_imgs = split["val"]
    val_masks = [mask_by[Path(i).stem] for i in val_imgs]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = np.arange(0.1, 0.91, 0.05)
    summary = {}
    for run in [CHECK_RUN] + RUNS:
        ckpt = LAB / "runs" / run / "checkpoint_best.pth"
        out = OUT / run
        if (out / "frozen_threshold.json").exists():
            raise SystemExit(f"{out} ja tem limiar congelado; nao sobrescrevo.")
        p, t, v = val_probabilities(ckpt, val_imgs, val_masks, device)
        thr = freeze_validation_threshold(p, t, v, out, thresholds=thresholds)
        frozen = json.loads((out / "frozen_threshold.json").read_text())
        frozen.update({"checkpoint": str(ckpt), "selected_on": "val (203 tiles)", "frozen_at": "2026-11-04"})
        (out / "frozen_threshold.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
        print(f"{run}: limiar {thr:.2f} (IoU val {frozen['validation_iou']:.4f})", flush=True)
        summary[run] = {"threshold": round(thr, 2), "val_iou": frozen["validation_iou"]}
        if run == CHECK_RUN:
            ref = json.loads((LAB / "runs" / run / "frozen_threshold.json").read_text())["threshold"]
            if abs(ref - thr) > 1e-6:
                raise SystemExit(f"Reimplementacao nao reproduz o limiar do run principal ({thr} != {ref}).")
            continue
        subprocess.run([sys.executable, str(LAB / "evaluate_unet_binary_test.py"), "--images", IMAGES,
                        "--masks", MASKS, "--split", SPLIT, "--checkpoint", str(ckpt),
                        "--threshold", str(out / "frozen_threshold.json"), "--output", str(out / "test_eval")],
                       check=True)
    (OUT / "resumo_validacao.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
