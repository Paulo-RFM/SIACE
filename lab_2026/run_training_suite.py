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


TEST_SCRIPT = "evaluate_unet_binary_test.py"
BINARY_SEGMENTATION_SCRIPTS = (
    "tree_segmentation_resnet.py",
    "tree_segmentation_resnet_26.py",
    "tree_segmentation_resnet_26_256softAug2.py",
    "tree_segmentation_resnet_26_512semAug.py",
    "tree_segmentation_resnet_26_512semAug2.py",
    "tree_segmentation_resnet_26_512softAug2.py",
    "tree_segmentation_resnet_26_512Aug5.py",
    "tree_segmentation_resnet_26_512softAug4.py",
)
MODERN_SEGMENTATION_SCRIPTS = {"tree_segmentation_resnet_26_512softAug4.py"}
EXPERIMENTS = {
    "tree_segmentation_resnet.py": {"resolution": 512, "batch": 6},
    "tree_segmentation_resnet_26.py": {"resolution": 512, "batch": 6},
    "tree_segmentation_resnet_26_256softAug2.py": {"resolution": 256, "batch": 12},
    "tree_segmentation_resnet_26_512semAug.py": {"resolution": 512, "batch": 6},
    "tree_segmentation_resnet_26_512semAug2.py": {"resolution": 512, "batch": 6},
    "tree_segmentation_resnet_26_512softAug2.py": {"resolution": 512, "batch": 6},
    "tree_segmentation_resnet_26_512softAug4.py": {"resolution": 512, "batch": 6},
}


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
    parser.add_argument(
        "--experiments", nargs="+", default=["all"], metavar="EXPERIMENT",
        help="all ou nomes de scripts; aceita a stem ou o nome completo.",
    )
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


def model_run_id(base_run_id, script_name):
    stem = Path(script_name).stem
    suffixes = {
        "tree_segmentation_resnet": "base",
        "tree_segmentation_resnet_26": "26",
    }
    suffix = suffixes.get(stem)
    if suffix is None:
        suffix = stem.removeprefix("tree_segmentation_resnet_26_")
        suffix = suffix.removeprefix("tree_segmentation_resnet_")
    return f"{base_run_id}_{suffix}"


def check_binary_scripts(base_dir):
    missing = [script for script in BINARY_SEGMENTATION_SCRIPTS
           if not (base_dir / script).is_file()]
    if missing:
        raise RuntimeError(f"Scripts binarios ausentes: {', '.join(missing)}")


def select_experiments(requested):
    if requested == ["all"]:
        return [script for script in BINARY_SEGMENTATION_SCRIPTS
                if script in EXPERIMENTS]
    selected = []
    aliases = {script: script for script in BINARY_SEGMENTATION_SCRIPTS}
    aliases.update({
        Path(script).stem.removeprefix("tree_segmentation_resnet_26_"): script
        for script in BINARY_SEGMENTATION_SCRIPTS
    })
    aliases.update({
        Path(script).stem.removeprefix("tree_segmentation_resnet_"): script
        for script in BINARY_SEGMENTATION_SCRIPTS
    })
    for requested_name in requested:
        script_name = aliases.get(requested_name)
        if script_name is None:
            raise ValueError(f"Experimento desconhecido: {requested_name}")
        if script_name not in EXPERIMENTS:
            raise ValueError(
                f"Experimento desativado ou nao adaptado: {requested_name}"
            )
        if script_name not in selected:
            selected.append(script_name)
    return selected


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    check_binary_scripts(base_dir)
    split_dir, images_dir, masks_dir, mapping = preflight(base_dir)
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.run_root)
    selected_scripts = select_experiments(args.experiments)

    env = os.environ.copy()
    env["UNET_SPATIAL_SPLIT_DIR"] = str(split_dir)
    env["UNET_IMAGES_DIR"] = str(images_dir)
    env["UNET_MASKS_DIR"] = str(masks_dir)
    selected = set(args.only or ("train", "test", "summary"))
    if args.skip_test:
        selected.discard("test")

    commands_by_model = {}
    for script_name in selected_scripts:
        model_run_id_value = model_run_id(run_id, script_name)
        run_dir = run_root / model_run_id_value
        if script_name in MODERN_SEGMENTATION_SCRIPTS:
            train_command = [
                sys.executable, "-u", str(base_dir / script_name),
                "--images-dir", str(images_dir), "--masks-dir", str(masks_dir),
                "--run-root", str(run_root), "--run-id", model_run_id_value,
            ]
        else:
            train_command = [
                sys.executable, "-u", str(base_dir / "run_legacy_binary_training.py"),
                "--script", str(base_dir / script_name),
                "--images-dir", str(images_dir), "--masks-dir", str(masks_dir),
                "--run-root", str(run_root), "--run-id", model_run_id_value,
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
            "--img-size", str(EXPERIMENTS[script_name]["resolution"]),
        ]
        commands_by_model[script_name] = {
            "run_id": model_run_id_value,
            "run_dir": str(run_dir),
            "train": train_command,
            "test": test_command,
        }

    if args.dry_run:
        commands = {}
        if "train" in selected:
            commands["train"] = {name: item["train"] for name, item in commands_by_model.items()}
        if "test" in selected:
            commands["test"] = {name: item["test"] for name, item in commands_by_model.items()}
        if "summary" in selected:
            commands["summary"] = ["build_run_index.py"]
        print(json.dumps({"run_id": run_id, "commands": commands}, indent=2))
        return

    for script_name, item in commands_by_model.items():
        run_dir = Path(item["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=False)
        log_file = run_dir / "log.txt"
        config = {
            "run_id": item["run_id"], "script": script_name, "architecture": "U-Net",
            "encoder": "resnet34", "seed": 42, "split_path": str(split_dir),
            "resolution": EXPERIMENTS[script_name]["resolution"],
            "batch": EXPERIMENTS[script_name]["batch"],
            "split_sha256": sha256(split_dir / "tiles_split.csv"),
            "partition_counts": mapping["split"].value_counts().to_dict(),
        }
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

        if "train" in selected:
            try:
                run_logged(item["train"], log_file, env, base_dir, "train")
            except Exception as exc:
                print(f"[ERRO] {script_name}: {exc}", file=sys.stderr)
                (run_dir / "failure.json").write_text(json.dumps({
                    "experiment": script_name,
                    "stage": "train",
                    "error": str(exc),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                continue

        checkpoint = run_dir / "checkpoint_best.pth"
        threshold = run_dir / "frozen_threshold.json"
        if "test" in selected:
            if not checkpoint.is_file():
                print(f"[ERRO] {script_name}: checkpoint ausente {checkpoint}", file=sys.stderr)
                continue
            if not threshold.is_file():
                print(f"[ERRO] {script_name}: limiar ausente {threshold}", file=sys.stderr)
                continue
            try:
                run_logged(item["test"], log_file, env, base_dir, "test")
            except Exception as exc:
                print(f"[ERRO] {script_name}: {exc}", file=sys.stderr)
                (run_dir / "failure.json").write_text(json.dumps({
                    "experiment": script_name,
                    "stage": "test",
                    "error": str(exc),
                }, ensure_ascii=False, indent=2), encoding="utf-8")

    if "summary" in selected:
        summary = {
            "run_id": run_id,
            "scripts": list(BINARY_SEGMENTATION_SCRIPTS),
            "split_sha256": sha256(split_dir / "tiles_split.csv"),
        }
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
