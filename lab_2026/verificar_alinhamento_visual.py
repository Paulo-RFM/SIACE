#!/usr/bin/env python3
"""Evidencia do desalinhamento antigo e verificacao visual do crop corrigido, em tiles reais do treino.

Para N tiles sorteados do treino e para cada faixa de escala usada no benchmark:
  * metodo ANTIGO (random.seed + dois RandomResizedCrop, como nos scripts originais): mede o
    deslocamento do canto do recorte e o IoU entre a mascara "certa" (com os parametros da imagem) e
    a mascara efetivamente usada;
  * metodo NOVO (aligned_augmentation.paired_random_resized_crop): mesmos indicadores (devem dar 0 px e 1,0).
Salva desalinhamento.csv, resumo.json e figuras com o contorno da mascara sobre a imagem (antigo × novo).
Saida: runs/alinhamento_auditoria_<data>/ (nao sobrescreve).
"""
import argparse
import json
import random
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage as ndi
from torchvision.transforms import InterpolationMode, RandomResizedCrop

from aligned_augmentation import paired_random_resized_crop

LAB = Path(__file__).resolve().parent
IMAGES, MASKS = Path("/home/paulomaciel/proj/imagens"), Path("/home/paulomaciel/proj/masks")
SPLIT = Path("/home/paulomaciel/proj/unet_spatial_split/tiles_split.csv")
CONFIGS = {"0.85-1.0 (softAug4, semAug2, softAug2)": ((0.85, 1.0), (3 / 4, 4 / 3)),
           "0.4-1.0 (base, _26)": ((0.4, 1.0), (1.0, 1.0)),
           "0.3-1.0 (512semAug)": ((0.3, 1.0), (1.0, 1.0))}
# Nota (2026-11-06): no 512semAug o bloco de augmentation esta dentro de uma string (codigo morto); a faixa
# 0,3-1,0 esta configurada mas nunca foi aplicada no treino. A linha fica como cenario hipotetico.


def binary(m):
    return np.any(np.array(m) != 0, axis=-1)


def iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def old_method(img, mask, scale, ratio):
    """Reproduz o codigo antigo; devolve tambem a mascara 'certa' (parametros da imagem, NEAREST)."""
    crop = RandomResizedCrop((512, 512), scale=scale, ratio=ratio)
    seed = np.random.randint(0, 100000)
    random.seed(seed)
    p_img = RandomResizedCrop.get_params(img, list(scale), list(ratio))
    im2 = TF.resized_crop(img, *p_img, [512, 512])
    random.seed(seed)
    p_mask = RandomResizedCrop.get_params(mask, list(scale), list(ratio))
    m_used = TF.resized_crop(mask, *p_mask, [512, 512])  # BILINEAR, padrao do RandomResizedCrop
    m_right = TF.resized_crop(mask, *p_img, [512, 512], interpolation=InterpolationMode.NEAREST)
    del crop
    shift = max(abs(p_img[0] - p_mask[0]), abs(p_img[1] - p_mask[1]))
    return im2, m_used, m_right, shift


def contour(ax, m, color):
    edge = ndi.binary_dilation(m, iterations=2) & ~ndi.binary_erosion(m, iterations=2)
    over = np.zeros(m.shape + (4,), float)
    over[edge] = matplotlib.colors.to_rgba(color)
    ax.imshow(over)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--figuras", type=int, default=6)
    ap.add_argument("--saida", default=str(LAB / "runs" / f"alinhamento_auditoria_{date.today():%Y%m%d}"))
    args = ap.parse_args()
    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=False)
    np.random.seed(42); random.seed(42); torch.manual_seed(42)
    split = pd.read_csv(SPLIT)
    tiles = sorted(split[split.split == "train"].tile.astype(str))
    tiles = [t for t in tiles if (IMAGES / f"{t}.png").exists() or (IMAGES / f"{t}.tif").exists()]
    sample = random.sample(tiles, min(args.n, len(tiles)))
    rows, figs = [], 0
    for t in sample:
        ip = next(p for p in [IMAGES / f"{t}.png", IMAGES / f"{t}.tif", IMAGES / f"{t}.jpg"] if p.exists())
        img = Image.open(ip).convert("RGB")
        mask = Image.open(MASKS / f"{t}.png").convert("RGB")
        for name, (scale, ratio) in CONFIGS.items():
            im_o, m_used, m_right, shift = old_method(img, mask, scale, ratio)
            im_n, m_new, params = paired_random_resized_crop(img, mask, (512, 512), scale, ratio)
            m_ref = TF.resized_crop(mask, *params, [512, 512], interpolation=InterpolationMode.NEAREST)
            rows.append({"tile": t, "config": name, "antigo_desloc_px": shift,
                         "antigo_iou": iou(binary(m_used), binary(m_right)),
                         "antigo_frac_mascara_dilatada": float(binary(m_used).sum() / max(binary(m_right).sum(), 1)),
                         "novo_iou": iou(binary(m_new), binary(m_ref))})
            if figs < args.figuras and name.startswith("0.85") and binary(m_right).any():
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                for ax, im, m, ttl in [(axs[0], im_o, binary(m_used), f"antigo (desloc. {shift} px)"),
                                       (axs[1], im_n, binary(m_new), "corrigido")]:
                    ax.imshow(im); contour(ax, m, "yellow"); ax.set_title(ttl); ax.axis("off")
                fig.suptitle(f"{t}: contorno da mascara usada no treino sobre a imagem recortada")
                fig.tight_layout(); fig.savefig(out / f"exemplo_{figs:02d}_{t}.png", dpi=110); plt.close(fig)
                figs += 1
    df = pd.DataFrame(rows)
    df.to_csv(out / "desalinhamento.csv", index=False)
    res = {}
    for name, g in df.groupby("config"):
        res[name] = {"tiles": len(g),
                     "antigo_desloc_px_mediana": float(g.antigo_desloc_px.median()),
                     "antigo_desloc_px_p90": float(g.antigo_desloc_px.quantile(0.9)),
                     "antigo_desloc_px_max": int(g.antigo_desloc_px.max()),
                     "antigo_iou_mediana": float(g.antigo_iou.median()),
                     "antigo_iou_p10": float(g.antigo_iou.quantile(0.1)),
                     "antigo_frac_iou_abaixo_0_7": float((g.antigo_iou < 0.7).mean()),
                     "antigo_razao_area_mascara_mediana": float(g.antigo_frac_mascara_dilatada.median()),
                     "novo_iou_min": float(g.novo_iou.min())}
    (out / "resumo.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
