#!/usr/bin/env python3
"""Indice dos runs de classificacao de especies -> lab_2026/index_classification.csv.

Uma linha por run com config.json contendo 'task' (EfficientNet ou U-Net multiclasse).
Valores vazios = artefato ausente (nunca estimado).
"""
import csv
import json
from pathlib import Path

import pandas as pd


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except json.JSONDecodeError:
        return {}


def main():
    base = Path(__file__).resolve().parent
    rows = []
    for cfg_path in sorted((base / "runs").glob("*/config.json")):
        cfg = read_json(cfg_path)
        if "task" not in cfg:
            continue
        run = cfg_path.parent
        suite = read_json(run / "suite_command.json")
        row = {
            "run_id": cfg.get("run_id", run.name), "experiment": suite.get("experiment", ""),
            "suite_run_id": suite.get("suite_run_id", ""), "script": cfg.get("script", ""),
            "task": cfg.get("task", ""), "architecture": cfg.get("architecture", ""),
            "backbone": cfg.get("backbone", cfg.get("encoder", "")), "variant": cfg.get("variant", ""),
            "smoke": cfg.get("smoke", ""), "split_sha256": (cfg.get("split") or {}).get("split_sha256", ""),
            "epochs_completed": cfg.get("epochs_completed", ""), "effective_epoch": cfg.get("effective_epoch", ""),
            "best_val": "", "test_accuracy": "", "test_balanced_accuracy": "", "test_f1_macro": "",
            "test_f1_weighted": "", "test_pixel_accuracy": "", "test_miou_fg": "", "test_n": "",
            "failed": (run / "failure.json").is_file(),
            "checkpoint": str(run / "checkpoint_best.pth") if (run / "checkpoint_best.pth").is_file() else "",
            "checkpoint_sha256": cfg.get("checkpoint_best_sha256", ""),
        }
        if cfg.get("best_val_f1_macro") is not None:
            row["best_val"] = f"val_f1_macro={cfg['best_val_f1_macro']}"
        elif (run / "metrics.csv").is_file():
            m = pd.read_csv(run / "metrics.csv")
            sel = cfg.get("selection_metric", "iou")
            if sel in m.columns and len(m):
                row["best_val"] = f"{sel}={m[sel].min() if sel == 'val_loss' else m[sel].max()}"
        t = read_json(run / "test_metrics_global.json")
        if t:
            row.update(test_accuracy=t.get("accuracy"), test_balanced_accuracy=t.get("balanced_accuracy"),
                       test_f1_macro=t.get("f1_macro"), test_f1_weighted=t.get("f1_weighted"),
                       test_n=t.get("number_of_samples"))
        t = read_json(run / "test_eval" / "test_metrics.json")
        if t:
            row.update(test_pixel_accuracy=t.get("pixel_accuracy"), test_miou_fg=t.get("miou_foreground_present"),
                       test_f1_macro=t.get("f1_macro_foreground_present"), test_n=t.get("n_test_tiles"))
        rows.append(row)
    out = base / "index_classification.csv"
    if rows:
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    print(f"Indice de classificacao: {out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
