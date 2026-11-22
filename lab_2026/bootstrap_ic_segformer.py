#!/usr/bin/env python3
"""IC 95% por bootstrap POR TILE para o sistema com a SegFormer MiT-B1 (2026-11-11).

Mesmos calculos, B = 2000, semente 42 e funcoes de bootstrap_ic_corrigido.py, agora sobre:
  - segmentacao na validacao: SegFormer x U-Net ResNet-34 (diferenca pareada; probabilidades de validacao
    recalculadas com os limiares ja congelados, pois nao estavam salvas);
  - deteccao de ponta a ponta no teste: h-maxima (metodo do sistema, escolhido na validacao) x componentes conexas
    x watershed por janela (runs/pipeline_test*_segformer_20261111) e x sistema anterior (U-Net + componentes
    conexas, pipeline_test_corrigido_20261106), pareados por tile;
  - classificacao de ponta a ponta nas arvores casadas (h-maxima).
O IoU/F1 de teste da segmentacao ja esta em runs/arquiteturas_comparacao_20261110/ic_95.csv.
Usa apenas predicoes ja salvas no teste. Saida: runs/bootstrap_ic_segformer_20261111/ic_95.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score

from bootstrap_ic_corrigido import IMAGES, MASKS, SEED, SPLIT, boot_idx, gt_valid, iou_f1, row
from unet_spatial_protocol import load_split, paired_tiles

LAB = Path(__file__).resolve().parent
R = LAB / "runs"
OUT = R / "bootstrap_ic_segformer_20261111"
SEGF = R / "arq_segformer_mit_b1_20261110"
UNET = R / "alinhado_20261105_2215_softAug4"
PT_HMAX = R / "pipeline_test_hmax_segformer_20261111"
PT_SEGF = R / "pipeline_test_segformer_20261111"
PT_WS = R / "pipeline_test_ws_segformer_20261111"
PT_UNET = R / "pipeline_test_corrigido_20261106"


def val_counts(run_dir, arch, enc, tiles, img_by, gt_by, valid_by):
    import segmentation_models_pytorch as smp
    import torch
    from torchvision import transforms
    thr = json.loads((run_dir / "frozen_threshold.json").read_text())["threshold"]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = getattr(smp, arch)(encoder_name=enc, encoder_weights=None, in_channels=3, classes=1).to(dev)
    m.load_state_dict(torch.load(run_dir / "checkpoint_best.pth", map_location=dev, weights_only=True))
    m.eval()
    out = np.zeros((len(tiles), 3), np.int64)
    with torch.no_grad():
        for k, t in enumerate(tiles):
            x = tf(Image.open(img_by[t]).convert("RGB")).unsqueeze(0).to(dev)
            p = torch.sigmoid(m(x))[0, 0].cpu().numpy() >= thr
            g, v = gt_by[t], valid_by[t]
            out[k] = [(p & g & v).sum(), (p & ~g & v).sum(), (~p & g & v).sum()]
    del m
    torch.cuda.empty_cache()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    pairs = paired_tiles(IMAGES, MASKS)
    split, _, _ = load_split(SPLIT, [i for i, _ in pairs])
    img_by = {Path(i).stem: i for i, _ in pairs}
    mask_by = {Path(i).stem: m for i, m in pairs}

    # ---------- segmentacao (validacao): SegFormer x U-Net ----------
    val = [Path(i).stem for i in split["val"]]
    gtv, vav = gt_valid(val, mask_by, img_by)
    cs = val_counts(SEGF, "Segformer", "mit_b1", val, img_by, gtv, vav)
    cu = val_counts(UNET, "Unet", "resnet34", val, img_by, gtv, vav)
    rng = np.random.default_rng(SEED)
    idxv = boot_idx(len(val), rng)
    for k, c in (("SegFormer MiT-B1", cs), ("U-Net ResNet-34", cu)):
        row(rows, f"segmentacao validacao: {k}", "IoU", iou_f1(c)[0], [iou_f1(c[i])[0] for i in idxv], len(val))
    row(rows, "segmentacao validacao: SegFormer - U-Net", "dif. IoU (pareada)", iou_f1(cs)[0] - iou_f1(cu)[0],
        [iou_f1(cs[i])[0] - iou_f1(cu[i])[0] for i in idxv], len(val))

    # ---------- deteccao de ponta a ponta ----------
    rd = lambda p: pd.read_csv(p / "avaliacao_teste" / "deteccao_por_tile.csv").set_index("tile")  # noqa: E731
    met = {"h-maxima (SegFormer, sistema)": rd(PT_HMAX), "componentes conexas (SegFormer)": rd(PT_SEGF),
           "watershed janela (SegFormer)": rd(PT_WS), "componentes conexas (U-Net, sistema anterior)": rd(PT_UNET)}
    tiles_d = sorted(set.intersection(*[set(d.index) for d in met.values()]))
    met = {k: d.loc[tiles_d] for k, d in met.items()}
    det = lambda d: 2 * d.tp.sum() / (d.n_pred.sum() + d.n_gt.sum())  # noqa: E731
    ratio = lambda d: d.n_pred.sum() / d.n_gt.sum()  # noqa: E731
    rng = np.random.default_rng(SEED)
    idxd = boot_idx(len(tiles_d), rng)
    smp_ = {k: np.array([det(d.iloc[i]) for i in idxd]) for k, d in met.items()}
    for k, d in met.items():
        row(rows, f"deteccao e2e teste: {k}", "F1", det(d), smp_[k], len(tiles_d))
        row(rows, f"deteccao e2e teste: {k}", "instancias por arvore", ratio(d), [ratio(d.iloc[i]) for i in idxd],
            len(tiles_d))
    base = "h-maxima (SegFormer, sistema)"
    for k in met:
        if k != base:
            row(rows, f"deteccao e2e teste: {base} - {k}", "dif. F1 (pareada)", det(met[base]) - det(met[k]),
                smp_[base] - smp_[k], len(tiles_d))

    # ---------- classificacao de ponta a ponta (arvores casadas, h-maxima) ----------
    ac = pd.read_csv(PT_HMAX / "avaliacao_teste" / "arvores_casadas.csv")
    ga = {t: d for t, d in ac.groupby("tile")}
    ta = sorted(ga)
    rng = np.random.default_rng(SEED)
    idxa = boot_idx(len(ta), rng)

    def e2e(d):
        y, p = d.true_class.to_numpy(), d.pred_class.to_numpy()
        return f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0), float((y == p).mean())
    s = np.array([e2e(pd.concat([ga[ta[j]] for j in i])) for i in idxa])
    est = e2e(ac)
    row(rows, "classificacao e2e teste (h-maxima, SegFormer)", "F1-macro", est[0], s[:, 0], len(ta), "tiles com >=1 arvore casada")
    row(rows, "classificacao e2e teste (h-maxima, SegFormer)", "acuracia", est[1], s[:, 1], len(ta), "tiles com >=1 arvore casada")

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "ic_95.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.drop(columns=["obs"]).round(3).to_string())


if __name__ == "__main__":
    main()
