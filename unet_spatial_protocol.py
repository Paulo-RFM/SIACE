#!/usr/bin/env python3
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import re
import socket
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import psutil
except ImportError:
    psutil = None
import torch
from PIL import Image


TILE_RE = re.compile(r"^OrtoRP-3cm_(\d{3})_(\d{3})$")


def tile_coordinates(path):
    match = TILE_RE.match(Path(path).stem)
    if not match:
        raise ValueError(f"Nome de tile invalido: {path}")
    return tuple(int(value) for value in match.groups())


def group_5x5(path):
    row, col = tile_coordinates(path)
    return f"r{(row - 1) // 5 + 1:02d}_c{(col - 1) // 5 + 1:02d}"


def paired_tiles(image_dir, mask_dir):
    mask_by_stem = {
        path.stem: path
        for path in Path(mask_dir).glob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".tif", ".jpg"}
    }
    pairs = []
    for image in sorted(Path(image_dir).glob("*.tif")):
        mask = mask_by_stem.get(image.stem)
        if mask is not None:
            pairs.append((str(image), str(mask)))
    return pairs


def is_fully_black(image_path, threshold=0):
    image = np.asarray(Image.open(image_path).convert("RGB"))
    return bool(np.all(image <= threshold))


def filter_black_tiles(pairs):
    usable = []
    excluded = []
    for image, mask in pairs:
        if is_fully_black(image):
            excluded.append(image)
        else:
            usable.append((image, mask))
    return usable, excluded


def load_split(split_dir, expected_images):
    split_dir = Path(split_dir)
    csv_path = split_dir / "tiles_split.csv"
    manifest_path = split_dir / "tiles_by_split.json"
    if not csv_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split persistido nao encontrado em {split_dir}. "
            "Execute prepare_unet_spatial_split.py antes do treino."
        )

    mapping = pd.read_csv(csv_path)
    if "tile" not in mapping.columns or "split" not in mapping.columns:
        raise ValueError(f"{csv_path} deve conter as colunas tile e split.")

    mapping = mapping.drop_duplicates("tile", keep=False)
    if mapping["tile"].duplicated().any():
        raise ValueError("O arquivo de split contem tiles duplicados.")
    if set(mapping["split"]) != {"train", "val", "test"}:
        raise ValueError("O arquivo de split deve conter train, val e test.")

    split = dict(zip(mapping["tile"], mapping["split"]))
    expected_names = {Path(path).stem: path for path in expected_images}
    missing = sorted(set(expected_names) - set(split))
    extra = sorted(set(split) - set(expected_names))
    if missing or extra:
        raise ValueError(
            f"O split nao corresponde aos tiles utilizaveis. "
            f"ausentes={len(missing)}, extras={len(extra)}."
        )

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    groups_by_split = manifest.get("groups", {})
    all_groups = []
    for split_name in ("train", "val", "test"):
        all_groups.extend(groups_by_split.get(split_name, []))
    if len(all_groups) != len(set(all_groups)):
        raise ValueError("Falha de integridade: um grupo espacial aparece em mais de uma particao.")

    result = {"train": [], "val": [], "test": []}
    for image in expected_images:
        result[split[Path(image).stem]].append(image)
    return result, mapping, manifest


def prepare_persistent_split(image_dir, mask_dir, split_dir, seed=42):
    pairs, excluded = filter_black_tiles(paired_tiles(image_dir, mask_dir))
    if not pairs:
        raise ValueError("Nenhum tile nao-preto foi encontrado.")

    by_group = {}
    for image, mask in pairs:
        by_group.setdefault(group_5x5(image), []).append((image, mask))

    groups = sorted(by_group)
    rng = random.Random(seed)
    rng.shuffle(groups)
    train_n = int(len(groups) * 0.60)
    val_n = int(len(groups) * 0.20)
    train_groups = groups[:train_n]
    val_groups = groups[train_n:train_n + val_n]
    test_groups = groups[train_n + val_n:]

    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    assignments = {
        "train": set(train_groups),
        "val": set(val_groups),
        "test": set(test_groups),
    }
    for image, mask in pairs:
        group = group_5x5(image)
        split_name = next(name for name, group_set in assignments.items() if group in group_set)
        rows.append({"tile": Path(image).stem, "split": split_name, "group": group})

    mapping = pd.DataFrame(rows).sort_values(["split", "tile"])
    mapping.to_csv(split_dir / "tiles_split.csv", index=False)
    manifest = {
        "seed": seed,
        "grouping": "5x5 spatial blocks",
        "groups": {name: sorted(values) for name, values in assignments.items()},
        "counts": {
            "train_tiles": int((mapping["split"] == "train").sum()),
            "val_tiles": int((mapping["split"] == "val").sum()),
            "test_tiles": int((mapping["split"] == "test").sum()),
            "excluded_black_tiles": len(excluded),
        },
    }
    (split_dir / "tiles_by_split.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_groups = sum((list(values) for values in assignments.values()), [])
    assert len(all_groups) == len(set(all_groups))
    print(
        f"Split espacial validado: train={manifest['counts']['train_tiles']} "
        f"val={manifest['counts']['val_tiles']} test={manifest['counts']['test_tiles']} "
        f"pretos_excluidos={len(excluded)}"
    )
    return split_dir / "tiles_split.csv", split_dir / "tiles_by_split.json"


def valid_pixel_mask(image):
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.any(arr != 0, axis=2)


def hash_files(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def runtime_metadata(config, run_dir, split_path, test_paths, run_id=None):
    run_id = run_id or uuid.uuid4().hex[:12]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    metadata = {
        "run_id": run_id,
        "seed": config["seed"],
        "configuration": config,
        "hardware": {
            "gpu": gpu,
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 2)
            if torch.cuda.is_available() else 0,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        },
        "environment": {
            "pytorch": torch.__version__,
            "python": platform.python_version(),
            "os": platform.platform(),
            "host": socket.gethostname(),
            "segmentation_models_pytorch": importlib.metadata.version(
                "segmentation-models-pytorch"
            ),
        },
        "split_path": str(split_path),
        "test_hash": hash_files(test_paths),
        "process_rss_mb": round(psutil.Process().memory_info().rss / 1024**2, 2)
        if psutil is not None else None,
        "gpu_mem_alloc_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2)
        if torch.cuda.is_available() else 0,
        "gpu_mem_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2)
        if torch.cuda.is_available() else 0,
    }
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(run_dir, f"run_{run_id}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def frozen_threshold(threshold_path, default=0.5):
    threshold_path = Path(threshold_path)
    if not threshold_path.is_file():
        return default
    return float(json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"])


def freeze_validation_threshold(probabilities, targets, valid_pixels, output_dir, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    probabilities = probabilities.reshape(-1)
    targets = targets.reshape(-1).astype(np.uint8)
    valid_pixels = valid_pixels.reshape(-1).astype(bool)
    probabilities = probabilities[valid_pixels]
    targets = targets[valid_pixels]

    rows = []
    best = None
    for threshold in thresholds:
        predicted = (probabilities >= threshold)
        tp = int(np.count_nonzero(predicted & (targets == 1)))
        fp = int(np.count_nonzero(predicted & (targets == 0)))
        fn = int(np.count_nonzero(~predicted & (targets == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        iou = tp / max(tp + fp + fn, 1)
        row = {"threshold": float(threshold), "iou": iou, "f1": f1,
               "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}
        rows.append(row)
        if best is None or (iou, threshold) > (best["iou"], best["threshold"]):
            best = row

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "threshold_iou_curve.csv", index=False)
    Path(output_dir / "frozen_threshold.json").write_text(
        json.dumps({"threshold": float(best["threshold"]), "criterion": "validation_iou",
                    "validation_iou": float(best["iou"])}, indent=2),
        encoding="utf-8",
    )
    return float(best["threshold"])
