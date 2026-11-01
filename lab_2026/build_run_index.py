#!/usr/bin/env python3
import csv
import json
from pathlib import Path


def main():
    base_dir = Path(__file__).resolve().parent
    runs_dir = base_dir / "runs"
    rows = []
    for config_path in sorted(runs_dir.glob("*/config.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        run_dir = config_path.parent
        metrics_path = run_dir / "metrics.csv"
        best_val_iou = ""
        if metrics_path.is_file():
            import pandas as pd
            metrics = pd.read_csv(metrics_path)
            if "iou" in metrics.columns and len(metrics):
                best_val_iou = metrics["iou"].max()

        threshold_path = run_dir / "frozen_threshold.json"
        threshold = ""
        if threshold_path.is_file():
            threshold = json.loads(threshold_path.read_text(encoding="utf-8")).get("threshold", "")
        test_path = run_dir / "test_eval" / "test_metrics.csv"
        test_iou = ""
        if test_path.is_file():
            import pandas as pd
            test_metrics = pd.read_csv(test_path)
            if "iou" in test_metrics.columns and len(test_metrics):
                test_iou = test_metrics.iloc[0]["iou"]

        rows.append({
            "run_id": config.get("run_id", run_dir.name),
            "data": config.get("run_start"),
            "script": config.get("script", config.get("configuration", {}).get("model", "")),
            "architecture": config.get("architecture", ""),
            "encoder": config.get("encoder", ""),
            "resolution": config.get("resolution", config.get("configuration", {}).get("img_size", "")),
            "batch": config.get("batch", config.get("configuration", {}).get("batch_size", "")),
            "lr": config.get("lr", config.get("configuration", {}).get("learning_rate", "")),
            "split": config.get("split_path", ""),
            "seed": config.get("seed", ""),
            "epochs_effective": "",
            "best_val_iou": best_val_iou,
            "test_iou": test_iou,
            "threshold": threshold,
            "checkpoint": str(run_dir / "checkpoint_best.pth"),
        })

    output = base_dir / "index.csv"
    fieldnames = list(rows[0].keys()) if rows else [
        "run_id", "data", "script", "architecture", "encoder", "resolution",
        "batch", "lr", "split", "seed", "epochs_effective", "best_val_iou",
        "test_iou", "threshold", "checkpoint",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Indice atualizado: {output} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
