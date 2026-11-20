#!/usr/bin/env python3
"""Avaliacao de ponta a ponta do pipeline nos tiles de TESTE (uma unica vez).

Entrada: saida de pipeline_cidade.py rodado com --partition test --save-masks.
  * Deteccao: instancias do watershed sobre a mascara PREDITA x poligonos anotados
    (label 'tree'; casamento um-para-um pelo algoritmo hungaro; TP se IoU >= 0,5).
  * Classificacao fim a fim: nas arvores casadas, classe predita x classe anotada
    (group_id ausente ou 11 -> classe 1, mesma regra do manifesto).
  * Regressao: compara a mascara binaria do pipeline com a do avaliador do P1.
Saida: <pipeline_dir>/avaliacao_teste/.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment

import species_data as sd
import species_metrics as sm

PROJ = sd.PROJECT_ROOT


def gt_polygons(tile):
    for d in ("labels", "labels2"):
        jp = PROJ / d / f"{tile}.json"
        if jp.is_file():
            out = []
            for sh in json.loads(jp.read_text(encoding="utf-8"))["shapes"]:
                if sh.get("label") != "tree" or len(sh.get("points", [])) < 3:
                    continue
                g = sh.get("group_id")
                cls = 1 if g is None or g == 11 else int(g)
                m = np.zeros((512, 512), np.uint8)
                cv2.fillPoly(m, [np.array(sh["points"], np.int32)], 1)
                out.append((m.astype(bool), cls))
            return out
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pipeline-dir", required=True)
    ap.add_argument("--p1-predictions", default=str(Path(__file__).resolve().parent / "runs" / "p1_20261101_140519"
                                                     / "test_eval" / "predictions_test"))
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    pdir = Path(args.pipeline_dir)
    out = pdir / "avaliacao_teste"
    if (out / "metricas_teste.json").exists():
        raise FileExistsError("Teste ja avaliado para este pipeline (avaliacao unica).")
    out.mkdir(exist_ok=True)
    prov = json.loads((pdir / "provenance.json").read_text())
    if prov.get("partition") != "test":
        raise ValueError("O pipeline informado nao foi rodado com --partition test.")
    class_ids, class_names = prov["class_ids"], prov["class_names"]
    inst_pred = pd.read_csv(pdir / "instancias.csv")
    pred_by = {(t, i): c for t, i, c in zip(inst_pred.tile, inst_pred.instance_id, inst_pred.pred_class)}

    tile_split, _ = sd.load_tile_split(sd.resolve_split_path())
    tiles = sorted(t for t, s in tile_split.items() if s == "test")
    rows, pairs, ious, reg_diff, reg_n = [], [], [], 0, 0
    for t in tiles:
        inst_path = pdir / "mascaras" / "instances" / f"{t}_instances.png"
        inst = np.array(Image.open(inst_path)).astype(np.int32) if inst_path.is_file() else np.zeros((512, 512), np.int32)
        p1 = Path(args.p1_predictions) / f"{t}.png"
        mine = pdir / "mascaras" / "binary" / f"{t}.png"
        if p1.is_file() and mine.is_file():
            reg_n += 1
            reg_diff += int(((np.array(Image.open(p1)) > 127) != (np.array(Image.open(mine)) > 127)).sum())
        gts = gt_polygons(t)
        ids = [int(i) for i in np.unique(inst) if i > 0]
        preds = [inst == i for i in ids]
        tp = 0
        if gts and preds:
            M = np.zeros((len(gts), len(preds)))
            for a, (g, _) in enumerate(gts):
                for b, p in enumerate(preds):
                    inter = (g & p).sum()
                    if inter:
                        M[a, b] = inter / (g | p).sum()
            r, c = linear_sum_assignment(-M)
            for a, b in zip(r, c):
                if M[a, b] >= args.iou:
                    tp += 1
                    ious.append(M[a, b])
                    pairs.append({"tile": t, "instance_id": ids[b], "iou": M[a, b], "true_class": gts[a][1],
                                  "pred_class": pred_by.get((t, ids[b]))})
        rows.append({"tile": t, "n_gt": len(gts), "n_pred": len(ids), "tp": tp})

    d = pd.DataFrame(rows)
    d.to_csv(out / "deteccao_por_tile.csv", index=False)
    TP, G, P = int(d.tp.sum()), int(d.n_gt.sum()), int(d.n_pred.sum())
    prec, rec = TP / max(P, 1), TP / max(G, 1)
    err = d.n_pred - d.n_gt
    det = {"iou_threshold": args.iou, "n_test_tiles": len(tiles), "n_gt_trees_per_tile": G, "n_instances": P,
           "tp": TP, "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
           "mean_iou_tp": float(np.mean(ious)) if ious else None,
           "count_error_mean_per_tile": float(err.mean()), "count_mae_per_tile": float(err.abs().mean()),
           "exact_count_tiles_pct": float((err == 0).mean() * 100), "instances_per_gt": P / max(G, 1)}

    pr = pd.DataFrame(pairs)
    pr.to_csv(out / "arvores_casadas.csv", index=False)
    pr = pr.dropna(subset=["pred_class"])
    idx = {c: k for k, c in enumerate(class_ids)}
    y = pr["true_class"].map(idx).to_numpy(); p = pr["pred_class"].astype(int).map(idx).to_numpy()
    labels = list(range(len(class_ids)))
    cls = sm.global_metrics(y, p, labels)
    sm.per_class_metrics(y, p, None, labels, class_names, class_ids).to_csv(out / "classificacao_por_classe.csv", index=False)
    (out / "classificacao_report.txt").write_text(sm.text_report(y, p, labels, class_names), encoding="utf-8")
    sm.save_confusion(y, p, labels, class_names, out, "fim_a_fim", "Pipeline fim a fim - teste")

    res = {"deteccao": det, "classificacao_nas_arvores_detectadas": cls,
           "regressao_mascara_vs_P1": {"tiles_comparados": reg_n, "pixels_diferentes": reg_diff},
           "pipeline": prov}
    sm.write_json(out / "metricas_teste.json", res)
    print(json.dumps({"deteccao": det, "classificacao": {k: cls[k] for k in ("accuracy", "balanced_accuracy",
                      "f1_macro", "number_of_samples")}, "regressao": res["regressao_mascara_vs_P1"]},
                     indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
