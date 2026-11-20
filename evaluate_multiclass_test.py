#!/usr/bin/env python3
"""Avaliacao UNICA no teste espacial da U-Net multiclasse (fundo + 6 classes de copa).

Metricas por pixel acumuladas sobre todos os tiles de teste (matriz de confusao
global, nao media de lotes):
  pixel_accuracy, mIoU/F1/precisao/recall macro (sem fundo), por classe
  (precision, recall, f1, iou, support), matriz de confusao (contagens e
  normalizada por linha/coluna, CSV+PNG), classification_report textual.

Saida em <run_dir>/test_eval/ (mesma convencao de evaluate_unet_binary_test.py).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from segmentation_models_pytorch import Unet

import species_data as sd
import species_metrics as sm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--split", required=True, help="diretorio ou tiles_split.csv")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--normalize", choices=["imagenet", "none"], required=True,
                   help="deve repetir a normalizacao usada no treino")
    p.add_argument("--species-names", default=str(Path(__file__).resolve().parent / "species_names.json"))
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--max-tiles", type=int, default=None, help="apenas para smoke")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)
    if (out / "test_metrics.json").exists():
        raise FileExistsError(f"Teste ja avaliado neste run: {out}. Avaliacao unica.")
    out.mkdir(parents=True, exist_ok=True)

    tile_split, split_info = sd.load_tile_split(sd.resolve_split_path(args.split))
    test_tiles = sorted(t for t, s in tile_split.items() if s == "test")
    images = [Path(args.images) / f"{t}.tif" for t in test_tiles]
    masks = [Path(args.masks) / f"{t}.png" for t in test_tiles]
    missing = [str(p) for p in images + masks if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} arquivos de teste ausentes, ex.: {missing[:3]}")
    if args.max_tiles:
        images, masks, test_tiles = images[:args.max_tiles], masks[:args.max_tiles], test_tiles[:args.max_tiles]

    names_map, _ = sd.load_species_names(args.species_names, range(1, 7))
    class_names = ["fundo"] + [names_map[i] for i in range(1, 7)]
    k = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=k).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    tf = [transforms.ToTensor()]
    if args.normalize == "imagenet":
        tf.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    tf = transforms.Compose(tf)
    size = (args.img_size, args.img_size)

    cm = np.zeros((k, k), dtype=np.int64)
    per_tile = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for tile, ip, mp in zip(test_tiles, images, masks):
            img = Image.open(ip).convert("RGB").resize(size, Image.BILINEAR)
            target = sd.rgb_mask_to_class(np.array(Image.open(mp).convert("RGB").resize(size, Image.NEAREST)))
            pred = model(tf(img).unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
            tile_cm = np.bincount(k * target.ravel().astype(np.int64) + pred.ravel(), minlength=k * k).reshape(k, k)
            cm += tile_cm
            per_tile.append({"tile": tile, **{f"gt_px_{class_names[c]}": int(tile_cm[c].sum()) for c in range(k)}})
    elapsed = time.perf_counter() - t0

    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
    support = cm.sum(1)
    by_class = pd.DataFrame({"class_index": range(k), "species_id": range(k), "class_name": class_names,
                             "precision": prec, "recall": rec, "f1": f1, "iou": iou, "support_px": support,
                             "present_in_test": support > 0})
    by_class.to_csv(out / "test_metrics_by_class.csv", index=False)
    fg = by_class[(by_class.class_index > 0) & by_class.present_in_test]
    metrics = {
        "pixel_accuracy": float(tp.sum() / cm.sum()),
        "miou_foreground_present": float(fg["iou"].mean()),
        "f1_macro_foreground_present": float(fg["f1"].mean()),
        "precision_macro_foreground_present": float(fg["precision"].mean()),
        "recall_macro_foreground_present": float(fg["recall"].mean()),
        "miou_all_classes": float(iou.mean()),
        "n_test_tiles": len(test_tiles), "classes_present": by_class.loc[by_class.present_in_test, "class_name"].tolist(),
        "split_sha256": split_info["split_sha256"],
        "test_hash": sd.sha256_text_lines(test_tiles),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sd.sha256_file(args.checkpoint),
        "normalize": args.normalize, "inference_total_s": elapsed,
        "inference_per_tile_s": elapsed / max(1, len(test_tiles)),
        "smoke_max_tiles": args.max_tiles,
    }
    sm.write_json(out / "test_metrics.json", metrics)
    pd.DataFrame([metrics]).drop(columns=["classes_present"]).to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(per_tile).to_csv(out / "test_pixels_per_tile.csv", index=False)

    # Matrizes (reaproveita o desenho de species_metrics a partir dos pares reconstruidos da cm)
    idx = [f"true:{n}" for n in class_names]; cols = [f"pred:{n}" for n in class_names]
    pd.DataFrame(cm, index=idx, columns=cols).to_csv(out / "test_confusion_counts.csv")
    with np.errstate(invalid="ignore", divide="ignore"):
        row = np.nan_to_num(cm / cm.sum(1, keepdims=True)); col = np.nan_to_num(cm / cm.sum(0, keepdims=True))
    pd.DataFrame(row, index=idx, columns=cols).to_csv(out / "test_confusion_row_normalized.csv")
    pd.DataFrame(col, index=idx, columns=cols).to_csv(out / "test_confusion_col_normalized.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for mat, suffix, fmt in ((row, "row_normalized", ".2f"), (col, "col_normalized", ".2f"), (cm / 1e3, "counts_kpx", ".0f")):
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)
        thr = mat.max() / 2 if mat.size else 0
        for i in range(k):
            for j in range(k):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] > thr else "black")
        ax.set_xlabel("Predito"); ax.set_ylabel("Referencia")
        ax.set_title(f"U-Net multiclasse - teste ({suffix})")
        fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
        fig.savefig(out / f"test_confusion_{suffix}.png", dpi=150); plt.close(fig)

    lines = [f"{'classe':<26}{'precision':>10}{'recall':>10}{'f1':>10}{'iou':>10}{'support_px':>14}"]
    for r in by_class.itertuples():
        lines.append(f"{r.class_name:<26}{r.precision:>10.4f}{r.recall:>10.4f}{r.f1:>10.4f}{r.iou:>10.4f}{r.support_px:>14d}")
    lines.append(f"\npixel_accuracy={metrics['pixel_accuracy']:.4f}  mIoU(fg presentes)={metrics['miou_foreground_present']:.4f}")
    (out / "test_classification_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Teste avaliado uma unica vez: {len(test_tiles)} tiles")
    print(json.dumps({k2: metrics[k2] for k2 in ("pixel_accuracy", "miou_foreground_present",
                                                 "f1_macro_foreground_present")}, indent=2))


if __name__ == "__main__":
    main()
