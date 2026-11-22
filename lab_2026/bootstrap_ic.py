#!/usr/bin/env python3
"""Intervalos de confianca de 95% por bootstrap POR TILE (P6 da revisao critica).

Os tiles sao a unidade de reamostragem (bootstrap por conglomerado): arvores e pixels de um mesmo
tile sao correlacionados, e reamostrar pixels ou copas isoladamente subestimaria a incerteza.
B = 2000 reamostras, semente 42, intervalo percentil. As diferencas entre metodos sao PAREADAS
(mesmos tiles reamostrados para os dois), o que responde se uma diferenca e distinguivel do acaso.

Usa apenas predicoes ja salvas no teste (nada e reavaliado no teste). Excecao: para comparar a
configuracao principal com a 512semAug NA VALIDACAO, as probabilidades de validacao dos dois
checkpoints sao recalculadas (com os limiares ja congelados), pois nao estavam salvas.
Saida: runs/bootstrap_ic_20261104/{ic_95.csv, ic_95.json}. Rodar a partir de lab_2026/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score

from unet_spatial_protocol import load_split, paired_tiles, valid_pixel_mask

LAB = Path(__file__).resolve().parent
R = LAB / "runs"
OUT = R / "bootstrap_ic_20261104"
IMAGES, MASKS, SPLIT = "/home/paulomaciel/proj/imagens", "/home/paulomaciel/proj/masks", "/home/paulomaciel/proj/unet_spatial_split"
B, SEED = 2000, 42
MAIN = "p1_20261101_140519"
SEG_RUNS = {  # nome -> pasta com test_eval/predictions_test (limiar escolhido na validacao)
    "principal (softAug4)": R / MAIN,
    "base": R / "limiar_val_herdados_20261104/20261101_185936_base",
    "26": R / "limiar_val_herdados_20261104/20261101_185936_26",
    "512semAug": R / "limiar_val_herdados_20261104/20261101_185936_512semAug",
    "512semAug2": R / "limiar_val_herdados_20261104/treino_20261101_203446_512semAug2",
    "512softAug2": R / "limiar_val_herdados_20261104/treino_20261101_203446_512softAug2",
}


def boot_idx(n, rng):
    return [rng.integers(0, n, n) for _ in range(B)]


def ci(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def row(rows, grupo, metrica, est, samples, n_tiles, obs=""):
    lo, hi = ci(samples)
    rows.append({"grupo": grupo, "metrica": metrica, "estimativa": est, "ic95_inf": lo, "ic95_sup": hi,
                 "n_tiles": n_tiles, "obs": obs})


def seg_counts(pred_png_dir, tiles, gt_by, valid_by):
    out = np.zeros((len(tiles), 3), np.int64)  # tp, fp, fn
    for k, t in enumerate(tiles):
        p = np.asarray(Image.open(pred_png_dir / f"{t}.png")) > 127
        g, v = gt_by[t], valid_by[t]
        out[k] = [(p & g & v).sum(), (p & ~g & v).sum(), (~p & g & v).sum()]
    return out


def iou_f1(c):
    tp, fp, fn = c.sum(0)
    return tp / (tp + fp + fn), 2 * tp / (2 * tp + fp + fn)


def gt_valid(tiles, mask_by, img_by):
    gt, va = {}, {}
    for t in tiles:
        gt[t] = np.any(np.asarray(Image.open(mask_by[t]).convert("RGB")) != 0, axis=2)
        va[t] = valid_pixel_mask(Image.open(img_by[t]))
    return gt, va


def val_counts(run_dir, threshold, tiles, img_by, gt_by, valid_by):
    import torch
    from segmentation_models_pytorch import Unet
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(dev)
    m.load_state_dict(torch.load(run_dir / "checkpoint_best.pth", map_location=dev, weights_only=True))
    m.eval()
    out = np.zeros((len(tiles), 3), np.int64)
    with torch.no_grad():
        for k, t in enumerate(tiles):
            x = tf(Image.open(img_by[t]).convert("RGB")).unsqueeze(0).to(dev)
            p = (torch.sigmoid(m(x))[0, 0].cpu().numpy() >= threshold)
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

    # ---------- segmentacao (teste) ----------
    test = [Path(i).stem for i in split["test"]]
    gt, va = gt_valid(test, mask_by, img_by)
    rng = np.random.default_rng(SEED)
    idx = boot_idx(len(test), rng)
    counts = {k: seg_counts(d / "test_eval" / "predictions_test", test, gt, va) for k, d in SEG_RUNS.items()}
    base = counts["principal (softAug4)"]
    for k, c in counts.items():
        iou, f1 = iou_f1(c)
        s = np.array([iou_f1(c[i]) for i in idx])
        row(rows, f"segmentacao teste: {k}", "IoU", iou, s[:, 0], len(test))
        row(rows, f"segmentacao teste: {k}", "F1", f1, s[:, 1], len(test))
        if k != "principal (softAug4)":
            d = [iou_f1(c[i])[0] - iou_f1(base[i])[0] for i in idx]
            row(rows, f"segmentacao teste: {k} - principal", "dif. IoU (pareada)", iou_f1(c)[0] - iou_f1(base)[0], d,
                len(test))

    # ---------- segmentacao (validacao): principal x 512semAug ----------
    val = [Path(i).stem for i in split["val"]]
    gtv, vav = gt_valid(val, mask_by, img_by)
    thr_main = json.loads((R / MAIN / "frozen_threshold.json").read_text())["threshold"]
    sem = R / "20261101_185936_512semAug"
    thr_sem = json.loads((SEG_RUNS["512semAug"] / "frozen_threshold.json").read_text())["threshold"]
    cm = val_counts(R / MAIN, thr_main, val, img_by, gtv, vav)
    cs = val_counts(sem, thr_sem, val, img_by, gtv, vav)
    rng = np.random.default_rng(SEED)
    idxv = boot_idx(len(val), rng)
    for k, c in (("principal (softAug4)", cm), ("512semAug", cs)):
        row(rows, f"segmentacao validacao: {k}", "IoU", iou_f1(c)[0], [iou_f1(c[i])[0] for i in idxv], len(val))
    row(rows, "segmentacao validacao: 512semAug - principal", "dif. IoU (pareada)", iou_f1(cs)[0] - iou_f1(cm)[0],
        [iou_f1(cs[i])[0] - iou_f1(cm[i])[0] for i in idxv], len(val))

    # ---------- classificador de especies (teste, recortes anotados) ----------
    cls = {"EfficientNet anotados": R / "classif_20261102_163825_efficientnet_anotados",
           "EfficientNet + aumento offline": R / "classif_20261102_163825_efficientnet_aug_treino"}
    preds = {k: pd.read_csv(d / "test_predictions.csv") for k, d in cls.items()}
    tiles_c = sorted(preds["EfficientNet anotados"].tile.unique())
    by_tile = {k: {t: g for t, g in p.groupby("tile")} for k, p in preds.items()}
    rng = np.random.default_rng(SEED)
    idxc = boot_idx(len(tiles_c), rng)

    def cls_metrics(df):
        y, p = df.species.to_numpy(), df.pred_species.to_numpy()
        labels = np.unique(y)
        return f1_score(y, p, labels=labels, average="macro", zero_division=0), balanced_accuracy_score(y, p)

    samples = {k: [] for k in cls}
    for i in idxc:
        for k in cls:
            samples[k].append(cls_metrics(pd.concat([by_tile[k][tiles_c[j]] for j in i])))
    for k in cls:
        est = cls_metrics(preds[k])
        s = np.array(samples[k])
        row(rows, f"classificador teste: {k}", "F1-macro", est[0], s[:, 0], len(tiles_c))
        row(rows, f"classificador teste: {k}", "acuracia balanceada", est[1], s[:, 1], len(tiles_c))
    a, b = np.array(samples["EfficientNet anotados"]), np.array(samples["EfficientNet + aumento offline"])
    ea, eb = cls_metrics(preds["EfficientNet anotados"]), cls_metrics(preds["EfficientNet + aumento offline"])
    row(rows, "classificador teste: anotados - aumento offline", "dif. F1-macro (pareada)", ea[0] - eb[0], a[:, 0] - b[:, 0],
        len(tiles_c))

    # ---------- bananeira: binario x multiclasse (um-contra-todos) ----------
    bin_dir = R / "classif_20261103_115814_efficientnet_binaria_bananeira"
    if (bin_dir / "test_predictions.csv").exists():
        mb = preds["EfficientNet anotados"].merge(pd.read_csv(bin_dir / "test_predictions.csv")[["crop_path", "pred_species"]],
                                                  on="crop_path", suffixes=("", "_bin"))
        mb["y"] = (mb.species == 5).astype(int)
        mb["pm"] = (mb.pred_species == 5).astype(int)
        mb["pb"] = (mb.pred_species_bin == 5).astype(int)
        g = {t: d for t, d in mb.groupby("tile")}
        f = lambda d, c: f1_score(d.y, d[c], zero_division=0)  # noqa: E731
        sm, sb = [], []
        for i in idxc:
            d = pd.concat([g[tiles_c[j]] for j in i])
            sm.append(f(d, "pm")); sb.append(f(d, "pb"))
        row(rows, "bananeira teste: multiclasse (um-contra-todos)", "F1", f(mb, "pm"), sm, len(tiles_c))
        row(rows, "bananeira teste: binario", "F1", f(mb, "pb"), sb, len(tiles_c))
        row(rows, "bananeira teste: binario - multiclasse", "dif. F1 (pareada)", f(mb, "pb") - f(mb, "pm"),
            np.array(sb) - np.array(sm), len(tiles_c))

    # ---------- deteccao de ponta a ponta: CC x Watershed ----------
    cc = pd.read_csv(R / "pipeline_test_cc_20261103_223209/avaliacao_teste/deteccao_por_tile.csv").set_index("tile")
    wsd = pd.read_csv(R / "pipeline_e2e_test_p1_20261102/per_tile.csv").set_index("tile")
    tiles_d = sorted(set(cc.index) & set(wsd.index))
    cc, wsd = cc.loc[tiles_d], wsd.loc[tiles_d]
    det = lambda d: 2 * d.tp.sum() / (d.n_pred.sum() + d.n_gt.sum())  # noqa: E731
    rng = np.random.default_rng(SEED)
    idxd = boot_idx(len(tiles_d), rng)
    scc = [det(cc.iloc[i]) for i in idxd]
    sws = [det(wsd.iloc[i]) for i in idxd]
    row(rows, "deteccao e2e teste: componentes conexas", "F1", det(cc), scc, len(tiles_d))
    row(rows, "deteccao e2e teste: watershed", "F1", det(wsd), sws, len(tiles_d))
    row(rows, "deteccao e2e teste: CC - watershed", "dif. F1 (pareada)", det(cc) - det(wsd), np.array(scc) - np.array(sws),
        len(tiles_d))
    ratio = lambda d: d.n_pred.sum() / d.n_gt.sum()  # noqa: E731
    row(rows, "deteccao e2e teste: componentes conexas", "instancias por arvore", ratio(cc), [ratio(cc.iloc[i]) for i in idxd],
        len(tiles_d))

    # ---------- classificacao de ponta a ponta (arvores casadas, CC) ----------
    ac = pd.read_csv(R / "pipeline_test_cc_20261103_223209/avaliacao_teste/arvores_casadas.csv")
    ga = {t: d for t, d in ac.groupby("tile")}
    ta = sorted(ga)
    rng = np.random.default_rng(SEED)
    idxa = boot_idx(len(ta), rng)

    def e2e(d):
        y, p = d.true_class.to_numpy(), d.pred_class.to_numpy()
        return f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0), float((y == p).mean())
    s = np.array([e2e(pd.concat([ga[ta[j]] for j in i])) for i in idxa])
    est = e2e(ac)
    row(rows, "classificacao e2e teste (CC)", "F1-macro", est[0], s[:, 0], len(ta), "tiles com >=1 arvore casada")
    row(rows, "classificacao e2e teste (CC)", "acuracia", est[1], s[:, 1], len(ta), "tiles com >=1 arvore casada")

    d = pd.DataFrame(rows)
    d.to_csv(OUT / "ic_95.csv", index=False)
    (OUT / "ic_95.json").write_text(json.dumps({"B": B, "seed": SEED, "metodo": "bootstrap percentil por tile",
                                                "resultados": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(d.drop(columns=["obs"]).round(3).to_string())


if __name__ == "__main__":
    main()
