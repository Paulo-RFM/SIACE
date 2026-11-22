#!/usr/bin/env python3
"""Compara o benchmark de segmentacao ANTES (crop desalinhado) e DEPOIS (crop sincronizado) da correcao.

Para cada uma das 6 configuracoes:
  1. run novo: limiar escolhido NA VALIDACAO (varredura 0,10-0,90, freeze_validation_threshold, o mesmo
     codigo do run principal) e congelado em arquivo antes do teste;
  2. teste avaliado uma vez com evaluate_unet_binary_test.py (mesmo script do baseline);
  3. comparacao pareada por tile no teste (bootstrap por tile, B=2000, seed 42) entre a predicao antiga
     (ja salva, com o limiar antigo escolhido na validacao) e a nova: IoU, Dice/F1, precisao, recall,
     acuracia e F1 de borda (tolerancias de 2 e 5 px, ~6 e ~16 cm), que mede a qualidade do contorno;
  4. tempo de treino (log.txt), epocas (metrics.csv) e tempo de inferencia (test_metrics.csv).
Nada e escrito nos runs antigos nem nos novos: saida em --saida (padrao runs/alinhamento_comparacao_<data>).
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from reavaliar_limiar_herdados import val_probabilities
from unet_spatial_protocol import freeze_validation_threshold, load_split, paired_tiles, valid_pixel_mask

LAB = Path(__file__).resolve().parent
R = LAB / "runs"
IMAGES, MASKS, SPLIT = "/home/paulomaciel/proj/imagens", "/home/paulomaciel/proj/masks", "/home/paulomaciel/proj/unet_spatial_split"
B, SEED = 2000, 42
OLD = {  # cfg -> (run antigo, pasta com frozen_threshold.json e test_eval/ do limiar escolhido na validacao)
    "softAug4": ("p1_20261101_140519", "p1_20261101_140519"),
    "512semAug": ("20261101_185936_512semAug", "limiar_val_herdados_20261104/20261101_185936_512semAug"),
    "26": ("20261101_185936_26", "limiar_val_herdados_20261104/20261101_185936_26"),
    "base": ("20261101_185936_base", "limiar_val_herdados_20261104/20261101_185936_base"),
    "512semAug2": ("treino_20261101_203446_512semAug2", "limiar_val_herdados_20261104/treino_20261101_203446_512semAug2"),
    "512softAug2": ("treino_20261101_203446_512softAug2", "limiar_val_herdados_20261104/treino_20261101_203446_512softAug2"),
}
TOLS = (2, 5)


def train_minutes(run_dir):
    if not (run_dir / "log.txt").exists():
        return float("nan")
    txt = (run_dir / "log.txt").read_text(errors="ignore")
    a = re.search(r"started_at=(\S+)", txt)
    b = re.search(r"ended_at=(\S+?)(?:\s|$)", txt)
    if not (a and b):
        return float("nan")
    return (datetime.fromisoformat(b.group(1)) - datetime.fromisoformat(a.group(1))).total_seconds() / 60


def boundary(m):
    return m & ~ndi.binary_erosion(m, border_value=1)


def tile_counts(pred, gt, valid):
    """tp, fp, fn, tn de pixel + contagens de borda (casadas dentro da tolerancia) por tolerancia."""
    p, g = pred & valid, gt & valid
    out = [int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum()), int((~p & ~g & valid).sum())]
    bp, bg = boundary(p) & valid, boundary(g) & valid
    dg = ndi.distance_transform_edt(~bg) if bg.any() else np.full(p.shape, np.inf)
    dp = ndi.distance_transform_edt(~bp) if bp.any() else np.full(p.shape, np.inf)
    for t in TOLS:
        out += [int((bp & (dg <= t)).sum()), int(bp.sum()), int((bg & (dp <= t)).sum()), int(bg.sum())]
    return out


COLS = ["tp", "fp", "fn", "tn"] + [f"{k}_{t}" for t in TOLS for k in ("bp_ok", "bp", "bg_ok", "bg")]


def metrics(c):
    s = c.sum(0)
    d = dict(zip(COLS, s))
    tp, fp, fn, tn = d["tp"], d["fp"], d["fn"], d["tn"]
    m = {"iou": tp / (tp + fp + fn), "dice_f1": 2 * tp / (2 * tp + fp + fn), "precision": tp / (tp + fp),
         "recall": tp / (tp + fn), "accuracy": (tp + tn) / (tp + fp + fn + tn)}
    for t in TOLS:
        bp_, br_ = d[f"bp_ok_{t}"] / max(d[f"bp_{t}"], 1), d[f"bg_ok_{t}"] / max(d[f"bg_{t}"], 1)
        m[f"borda_f1_{t}px"] = 2 * bp_ * br_ / max(bp_ + br_, 1e-12)
    return m


def counts_from_pngs(pred_dir, tiles, gt_by, valid_by):
    return np.array([tile_counts(np.asarray(Image.open(pred_dir / f"{t}.png")) > 127, gt_by[t], valid_by[t])
                     for t in tiles], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stamp", required=True, help="carimbo dos runs novos: runs/alinhado_<stamp>_<cfg>")
    ap.add_argument("--saida", default=str(R / f"alinhamento_comparacao_{date.today():%Y%m%d}"))
    args = ap.parse_args()
    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)

    pairs = paired_tiles(IMAGES, MASKS)
    split, _, _ = load_split(SPLIT, [i for i, _ in pairs])
    mask_by = {Path(i).stem: m for i, m in pairs}
    img_by = {Path(i).stem: i for i, _ in pairs}
    val_imgs = split["val"]
    val_masks = [mask_by[Path(i).stem] for i in val_imgs]
    test_tiles = sorted(Path(i).stem for i in split["test"])
    gt_by = {t: np.any(np.asarray(Image.open(mask_by[t]).convert("RGB")) != 0, axis=2) for t in test_tiles}
    valid_by = {t: valid_pixel_mask(Image.open(img_by[t])) for t in test_tiles}
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = np.arange(0.1, 0.91, 0.05)

    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(test_tiles), len(test_tiles)) for _ in range(B)]
    rows, tiles_rows = [], []
    for cfg, (old_run, old_eval) in OLD.items():
        new_run = R / f"alinhado_{args.stamp}_{cfg}"
        if not (new_run / "checkpoint_best.pth").exists():
            print(f"{cfg}: {new_run.name} sem checkpoint, pulando", flush=True)
            continue
        o = out / cfg
        if not (o / "frozen_threshold.json").exists():
            p, t, v = val_probabilities(new_run / "checkpoint_best.pth", val_imgs, val_masks, device)
            freeze_validation_threshold(p, t, v, o, thresholds=thresholds)
            fr = json.loads((o / "frozen_threshold.json").read_text())
            fr.update({"checkpoint": str(new_run / "checkpoint_best.pth"), "selected_on": "val (203 tiles)",
                       "frozen_at": str(date.today())})
            (o / "frozen_threshold.json").write_text(json.dumps(fr, indent=2), encoding="utf-8")
        if not (o / "test_eval" / "test_metrics.csv").exists():
            subprocess.run([sys.executable, str(LAB / "evaluate_unet_binary_test.py"), "--images", IMAGES,
                            "--masks", MASKS, "--split", SPLIT, "--checkpoint", str(new_run / "checkpoint_best.pth"),
                            "--threshold", str(o / "frozen_threshold.json"), "--output", str(o / "test_eval")],
                           check=True)
        fr_new = json.loads((o / "frozen_threshold.json").read_text())
        fr_old = json.loads((R / old_eval / "frozen_threshold.json").read_text())
        tm_new = pd.read_csv(o / "test_eval" / "test_metrics.csv").iloc[0]
        tm_old = pd.read_csv(R / old_eval / "test_eval" / "test_metrics.csv").iloc[0]
        c_old = counts_from_pngs(R / old_eval / "test_eval" / "predictions_test", test_tiles, gt_by, valid_by)
        c_new = counts_from_pngs(o / "test_eval" / "predictions_test", test_tiles, gt_by, valid_by)
        m_old, m_new = metrics(c_old), metrics(c_new)
        boots = [(metrics(c_old[i]), metrics(c_new[i])) for i in idx]
        for k in m_old:
            so = np.array([b[0][k] for b in boots]); sn = np.array([b[1][k] for b in boots])
            dif = sn - so
            rows.append({"config": cfg, "metrica": k, "antigo": m_old[k], "novo": m_new[k],
                         "antigo_ic95": f"[{np.percentile(so, 2.5):.3f}; {np.percentile(so, 97.5):.3f}]",
                         "novo_ic95": f"[{np.percentile(sn, 2.5):.3f}; {np.percentile(sn, 97.5):.3f}]",
                         "dif_abs": m_new[k] - m_old[k], "dif_rel_pct": 100 * (m_new[k] - m_old[k]) / m_old[k],
                         "dif_ic95_inf": float(np.percentile(dif, 2.5)), "dif_ic95_sup": float(np.percentile(dif, 97.5)),
                         "frac_boot_novo_melhor": float((dif > 0).mean())})
        rows.append({"config": cfg, "metrica": "limiar_val", "antigo": fr_old["threshold"], "novo": fr_new["threshold"]})
        rows.append({"config": cfg, "metrica": "iou_val", "antigo": fr_old.get("validation_iou"), "novo": fr_new.get("validation_iou")})
        rows.append({"config": cfg, "metrica": "treino_min", "antigo": train_minutes(R / old_run), "novo": train_minutes(new_run)})
        rows.append({"config": cfg, "metrica": "epocas", "antigo": len(pd.read_csv(R / old_run / "metrics.csv")),
                     "novo": len(pd.read_csv(new_run / "metrics.csv"))})
        rows.append({"config": cfg, "metrica": "inferencia_s_por_tile", "antigo": tm_old.inference_per_tile_s,
                     "novo": tm_new.inference_per_tile_s})
        for t, a, b in zip(test_tiles, c_old, c_new):
            tiles_rows.append({"config": cfg, "tile": t, **{f"antigo_{k}": v for k, v in zip(COLS, a)},
                               **{f"novo_{k}": v for k, v in zip(COLS, b)}})
        print(f"{cfg}: IoU teste {m_old['iou']:.4f} -> {m_new['iou']:.4f}; limiar {fr_old['threshold']:.2f} -> {fr_new['threshold']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "comparacao.csv", index=False)
    pd.DataFrame(tiles_rows).to_csv(out / "contagens_por_tile.csv", index=False)
    (out / "comparacao.json").write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    # grafico IoU e F1 de borda, antigo x novo, com IC95
    ok = df[df.metrica.isin(["iou", "borda_f1_2px"])].copy()
    if len(ok):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, met, ttl in [(axs[0], "iou", "IoU de teste"), (axs[1], "borda_f1_2px", "F1 de borda (2 px) no teste")]:
            d = ok[ok.metrica == met].reset_index(drop=True)
            x = np.arange(len(d))
            for off, col, lab, ic in [(-0.2, "#9aa5b1", "com desalinhamento", "antigo_ic95"),
                                      (0.2, "#2f6f4f", "corrigido", "novo_ic95")]:
                lo = [float(s.strip("[]").split(";")[0]) for s in d[ic]]
                hi = [float(s.strip("[]").split(";")[1]) for s in d[ic]]
                v = d["antigo" if off < 0 else "novo"]
                ax.bar(x + off, v, 0.38, color=col, label=lab,
                       yerr=[v - np.array(lo), np.array(hi) - v], capsize=3)
            ax.set_xticks(x); ax.set_xticklabels(d.config, rotation=20)
            lo_all = min(d.antigo.min(), d.novo.min())
            ax.set_ylim(max(0, lo_all - 0.1), min(1, max(d.antigo.max(), d.novo.max()) + 0.06))
            ax.set_title(ttl); ax.grid(axis="y", alpha=0.3)
        axs[0].legend(loc="upper left", fontsize=8)
        fig.tight_layout(); fig.savefig(out / "iou_borda_antigo_vs_corrigido.png", dpi=150); plt.close(fig)
    print(df[df.metrica.isin(["iou", "dice_f1", "borda_f1_2px"])][["config", "metrica", "antigo", "novo", "dif_abs",
                                                                  "dif_ic95_inf", "dif_ic95_sup"]].to_string())


if __name__ == "__main__":
    main()
