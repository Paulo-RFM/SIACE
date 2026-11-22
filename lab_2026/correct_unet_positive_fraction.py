#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def valid_pixel_mask(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.any(rgb != 0, axis=2)


def positive_fraction(image_paths, mask_by_tile):
    foreground = 0
    valid_pixels = 0
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        mask_path = mask_by_tile[Path(image_path).stem]
        label = np.any(np.asarray(Image.open(mask_path).convert("RGB")) != 0, axis=2)
        valid = valid_pixel_mask(image)
        foreground += int(label[valid].sum())
        valid_pixels += int(valid.sum())
    return foreground / max(valid_pixels, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--output", help="Padrao: <run-dir>/config_corrigido.json")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    split = pd.read_csv(Path(args.split_dir) / "tiles_split.csv")
    images = {path.stem: path for path in Path(args.images_dir).glob("*.tif")}
    masks = {
        path.stem: path
        for path in Path(args.masks_dir).iterdir()
        if path.is_file()
    }
    fractions = {}
    for split_name in ("train", "val", "test"):
        tiles = split.loc[split["split"] == split_name, "tile"].astype(str)
        missing = [tile for tile in tiles if tile not in images or tile not in masks]
        if missing:
            raise ValueError(f"Tiles ausentes em {split_name}: {len(missing)}")
        fractions[split_name] = positive_fraction(
            [images[tile] for tile in tiles],
            masks,
        )

    source = run_dir / "config.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["positive_fraction_by_split"] = {
        name: float(fraction)
        for name, fraction in fractions.items()
    }
    output = Path(args.output) if args.output else run_dir / "config_corrigido.json"
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(fractions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
