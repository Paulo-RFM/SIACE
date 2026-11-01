#!/usr/bin/env python3

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SEGMENTATION_SCRIPT = "tree_segmentation_resnet_26_512softAug4.py"
TEST_SCRIPT = "evaluate_unet_binary_test.py"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(base_dir):
    split_dir = Path(os.environ.get(
        "UNET_SPATIAL_SPLIT_DIR",
        "/home/paulomaciel/proj/unet_spatial_split",
    ))
    split_csv = split_dir / "tiles_split.csv"
    if not split_csv.is_file():
        raise RuntimeError(f"Pre-voo falhou: split ausente em {split_csv}")

    mapping = __import__("pandas").read_csv(split_csv)
    images_dir = Path(os.environ.get(
        "UNET_IMAGES_DIR",
        "/home/paulomaciel/proj/imagens",
    ))
    masks_dir = Path(os.environ.get(
        "UNET_MASKS_DIR",
        "/home/paulomaciel/proj/masks",
    ))
    if not images_dir.is_dir():
        raise RuntimeError(f"Pre-voo falhou: pasta de imagens ausente: {images_dir}")
    if not masks_dir.is_dir():
        raise RuntimeError(f"Pre-voo falhou: pasta de mascaras ausente: {masks_dir}")

    image_files = list(images_dir.iterdir())
    if len(image_files) != 1018:
        raise RuntimeError(
            f"Pre-voo falhou: esperados 1018 arquivos na raiz, encontrados {len(image_files)}."
        )
    image_names = {p.stem for p in image_files if p.is_file()}
    mask_names = {p.stem for p in masks_dir.iterdir() if p.is_file()}
    if image_names != mask_names:
        raise RuntimeError(
            "Pre-voo falhou: casamento 1:1 imagem/mascara divergente "
            f"(imagens={len(image_names)}, mascaras={len(mask_names)})."
        )
    if len(mapping) != 1018:
        raise RuntimeError(f"Pre-voo falhou: split tem {len(mapping)} linhas, esperadas 1018.")

    stale_threshold = split_dir / "frozen_threshold.json"
    if stale_threshold.is_file():
        raise RuntimeError(
            f"Pre-voo falhou: limiar antigo encontrado em {stale_threshold}. "
            "Remova-o ou confirme que pertence ao run atual; o limiar e artefato do run."
        )

    usage = shutil.disk_usage(base_dir)
    required_gb = 20 * 1024**3
    if usage.free < required_gb:
        raise RuntimeError(
            f"Pre-voo falhou: apenas {usage.free / 1024**3:.1f} GiB livres; "
            "sao recomendados pelo menos 20 GiB."
        )
    return split_dir, images_dir, masks_dir, mapping


def parse_args():
    parser = argparse.ArgumentParser(description="Orquestra segmentacao binaria U-Net.")
    parser.add_argument("--dry-run", action="store_true", help="Somente lista os comandos.")
    parser.add_argument("--only", choices=("train", "test", "summary"), action="append")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--run-root", default="/home/paulomaciel/proj/lab_2026/runs")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def run_logged(command, log_file, env, cwd, label):
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    started = time.perf_counter()
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n[{label}] command={json.dumps(command)}\n")
        log.write(f"[{label}] started_at={started_at}\n")
        log.flush()
        try:
            subprocess.run(command, cwd=cwd, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            log.write(f"[{label}] failed code={exc.returncode}\n")
            raise RuntimeError(f"Etapa {label} falhou com codigo {exc.returncode}") from exc
    ended_at = dt.datetime.now(dt.timezone.utc).isoformat()
    elapsed = time.perf_counter() - started
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"[{label}] ended_at={ended_at}\n")
        log.write(f"[{label}] elapsed_s={elapsed:.3f}\n")


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    split_dir, images_dir, masks_dir, mapping = preflight(base_dir)
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.run_root)
    run_dir = run_root / run_id

    config = {
        "run_id": run_id, "script": SEGMENTATION_SCRIPT, "architecture": "U-Net",
        "encoder": "resnet34", "resolution": 512, "batch": 6, "lr": 1e-4,
        "seed": 42, "split_path": str(split_dir),
        "split_sha256": sha256(split_dir / "tiles_split.csv"),
        "partition_counts": mapping["split"].value_counts().to_dict(),
    }

    env = os.environ.copy()
    env["UNET_SPATIAL_SPLIT_DIR"] = str(split_dir)
    env["UNET_IMAGES_DIR"] = str(images_dir)
    env["UNET_MASKS_DIR"] = str(masks_dir)
    selected = set(args.only or ("train", "test", "summary"))
    if args.skip_test:
        selected.discard("test")

    train_command = [
        sys.executable, "-u", str(base_dir / SEGMENTATION_SCRIPT),
        "--images-dir", str(images_dir), "--masks-dir", str(masks_dir),
        "--run-root", str(run_root), "--run-id", run_id,
    ]
    if args.smoke:
        train_command.append("--smoke")

    checkpoint = run_dir / "checkpoint_best.pth"
    threshold = run_dir / "frozen_threshold.json"
    test_command = [
        sys.executable, "-u", str(base_dir / TEST_SCRIPT),
        "--images", str(images_dir), "--masks", str(masks_dir),
        "--split", str(split_dir), "--checkpoint", str(checkpoint),
        "--threshold", str(threshold), "--output", str(run_dir / "test_eval"),
        "--img-size", "512",
    ]

    if args.dry_run:
        commands = {}
        if "train" in selected:
            commands["train"] = train_command
        if "test" in selected:
            commands["test"] = test_command
        if "summary" in selected:
            commands["summary"] = ["build_run_index.py"]
        print(json.dumps({"run_id": run_id, "commands": commands}, indent=2))
        return

    run_dir.mkdir(parents=True, exist_ok=False)
    log_file = run_dir / "log.txt"
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    if "train" in selected:
        run_logged(train_command, log_file, env, base_dir, "train")

    if "test" in selected:
        if not checkpoint.is_file():
            raise RuntimeError(f"Teste abortado: checkpoint do run ausente: {checkpoint}")
        if not threshold.is_file():
            raise RuntimeError(f"Teste abortado: limiar do run ausente: {threshold}")
        run_logged(test_command, log_file, env, base_dir, "test")

    if "summary" in selected:
        summary = dict(config)
        summary["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        subprocess.run(
            [sys.executable, str(base_dir / "build_run_index.py")],
            cwd=base_dir,
            check=True,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
