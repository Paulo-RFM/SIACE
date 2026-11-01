#!/usr/bin/env python3
import argparse
import json
import sys
import os
import random
import re
from pathlib import Path

import pandas as pd
import sklearn.model_selection
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    script = Path(args.script).resolve()
    base_dir = script.parent
    split_dir = Path(__import__("os").environ.get(
        "UNET_SPATIAL_SPLIT_DIR",
        "/home/paulomaciel/proj/unet_spatial_split",
    ))
    run_dir = Path(args.run_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(split_dir / "tiles_split.csv")
    split_by_tile = dict(zip(mapping["tile"].astype(str), mapping["split"].astype(str)))

    def spatial_train_test_split(first, second=None, *positional, **kwargs):
        first_items = list(first)
        second_items = None if second is None else list(second)
        indices_by_split = {"train": [], "val": []}
        for index, item in enumerate(first_items):
            tile = Path(str(item)).stem
            split_name = split_by_tile.get(tile)
            if split_name in indices_by_split:
                indices_by_split[split_name].append(index)

        def select(items, indices):
            return None if items is None else [items[index] for index in indices]

        train_indices = indices_by_split["train"]
        val_indices = indices_by_split["val"]
        if second is None:
            return [first_items[index] for index in train_indices], \
                [first_items[index] for index in val_indices]
        return (
            [first_items[index] for index in train_indices],
            [first_items[index] for index in val_indices],
            select(second_items, train_indices),
            select(second_items, val_indices),
        )

    sklearn.model_selection.train_test_split = spatial_train_test_split

    checkpoint_path = run_dir / "checkpoint_best.pth"
    original_torch_save = torch.save

    def save_to_run_dir(obj, destination, *save_args, **save_kwargs):
        destination_name = Path(str(destination)).name
        if destination_name.endswith(".pth") and isinstance(obj, dict):
            original_torch_save(obj, checkpoint_path, *save_args, **save_kwargs)
            return
        return original_torch_save(obj, destination, *save_args, **save_kwargs)

    torch.save = save_to_run_dir
    original_to_csv = pd.DataFrame.to_csv

    def run_metrics_csv(self, destination=None, *csv_args, **csv_kwargs):
        destination = run_dir / "metrics.csv"
        return original_to_csv(self, destination, *csv_args, **csv_kwargs)

    pd.DataFrame.to_csv = run_metrics_csv
    sys.path.insert(0, str(base_dir))
    try:
        source = script.read_text(encoding="utf-8")
        source = re.sub(
            r'^base_image_tiles_dir\s*=.*$',
            f'base_image_tiles_dir = {str(args.images_dir)!r}',
            source,
            flags=re.MULTILINE,
        )
        source = re.sub(
            r'^base_mask_tiles_dir\s*=.*$',
            f'base_mask_tiles_dir = {str(args.masks_dir)!r}',
            source,
            flags=re.MULTILINE,
        )
        source = re.sub(
            r'^checkpoint_path\s*=.*$',
            f'checkpoint_path = {str(checkpoint_path)!r}',
            source,
            flags=re.MULTILINE,
        )
        if args.smoke:
            source = re.sub(r'^epochs\s*=.*$', 'epochs = 1', source, count=1, flags=re.MULTILINE)
        namespace = {"__name__": "__main__", "__file__": str(script)}
        exec(compile(source, str(script), "exec"), namespace, namespace)
    finally:
        torch.save = original_torch_save
        pd.DataFrame.to_csv = original_to_csv

    if not checkpoint_path.is_file():
        raise RuntimeError(f"Modelo legado nao salvou checkpoint: {script.name}")
    (run_dir / "frozen_threshold.json").write_text(json.dumps({
        "threshold": 0.5,
        "criterion": "adapter_default",
    }), encoding="utf-8")
    print(f"Modelo legado treinado com split espacial: {script.name}")


if __name__ == "__main__":
    main()
