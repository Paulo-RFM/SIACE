#!/usr/bin/env python3
"""Orquestra todos os treinos de CLASSIFICACAO de especies (analogo a run_training_suite.py).

Experimentos (cada um vira runs/<run_id>_<sufixo>/):
  efficientnet_anotados   EfficientNet-B0, recortes anotados; aumento so on-the-fly no treino
  efficientnet_aug_treino EfficientNet-B0 + copias aumentadas offline SOMENTE do treino
  unet_mc_ce              U-Net multiclasse por pixel (resnet2.py: CE sem pesos)
  unet_mc_ce_dice         U-Net multiclasse por pixel (resnet3.py: CE ponderada + Dice)
  unet_mc_ce_dice_initP1  resnet3.py inicializado com runs/p1_20261101_140519/checkpoint_best.pth

Desativados (motivo registrado em DISABLED): pseudorrotulos (sem CSV valido) e
catboost_final.py (outro problema: dados tabulares 'attack', nao especies).

Todos usam o mesmo split espacial (tiles_split.csv). O teste e avaliado uma unica
vez por run: dentro de train_species_efficientnet.py (EfficientNet) ou por
evaluate_multiclass_test.py (U-Net multiclasse). Ao final gera index_classification.csv.

Uso:
  python run_classification_suite.py --dry-run
  python run_classification_suite.py --smoke --experiments efficientnet_anotados
  python run_classification_suite.py                     # todos
  python run_classification_suite.py --only test --run-id <id_existente>   # reavalia checkpoints ja treinados
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import species_data as sd

BASE_DIR = Path(__file__).resolve().parent
PROJ = sd.PROJECT_ROOT
P1_CHECKPOINT = BASE_DIR / "runs" / "p1_20261101_140519" / "checkpoint_best.pth"
DEFAULTS = {
    "manifest": PROJ / "species_dataset_v2" / "species_manifest.csv",
    "species_names": BASE_DIR / "species_names.json",
    "images": PROJ / "imagens",
    "multiclass_masks": BASE_DIR / "masks_distin2",
}

AUGMENT_TARGET = 300

EXPERIMENTS = {
    "efficientnet_anotados": {
        "kind": "efficientnet", "script": "train_species_efficientnet.py", "augment": False,
        "description": "EfficientNet-B0, somente anotados",
    },
    "efficientnet_aug_treino": {
        "kind": "efficientnet", "script": "train_species_efficientnet.py", "augment": True,
        "description": "EfficientNet-B0, anotados + aumento offline apenas do treino",
    },
    "unet_mc_ce": {
        "kind": "multiclass", "script": "resnet2.py", "normalize": "none", "init": None,
        "description": "U-Net/ResNet-34 multiclasse, CrossEntropy",
    },
    "unet_mc_ce_dice": {
        "kind": "multiclass", "script": "resnet3.py", "normalize": "imagenet", "init": None,
        "description": "U-Net/ResNet-34 multiclasse, CE ponderada + Dice",
    },
    "unet_mc_ce_dice_initP1": {
        "kind": "multiclass", "script": "resnet3.py", "normalize": "imagenet", "init": str(P1_CHECKPOINT),
        "description": "Igual a unet_mc_ce_dice, inicializada com o checkpoint binario do run P1",
    },
}
DISABLED = {
    "efficientnet_pseudo": "Sem CSV de pseudorrotulos com proveniencia valida (dataset_auto/* rejeitados).",
    "catboost_final.py": "Script de outro problema (features f0..f523, alvo 'attack'); nao classifica especies.",
}


def sha256(path):
    return sd.sha256_file(path)


def preflight(args, selected):
    problems = []
    split_path = sd.resolve_split_path(args.split)
    tile_split, split_info = sd.load_tile_split(split_path)
    if not Path(args.species_names).is_file():
        problems.append(f"species_names ausente: {args.species_names}")
    else:
        sd.load_species_names(args.species_names, range(1, 7))

    kinds = {EXPERIMENTS[e]["kind"] for e in selected}
    if "efficientnet" in kinds:
        import pandas as pd
        if not Path(args.manifest).is_file():
            problems.append(f"manifesto ausente: {args.manifest} (rode build_species_manifest.py)")
        else:
            manifest = sd.validate_manifest(pd.read_csv(args.manifest), tile_split)
            elig = sd.test_eligibility(manifest)
            if not elig["meaningful_test"]:
                problems.append("nenhuma especie com >= 10 amostras de teste")
            missing = [p for p in manifest["crop_path"] if not Path(p).is_file()]
            if missing:
                problems.append(f"{len(missing)} recortes ausentes")
    if "multiclass" in kinds:
        images = {p.stem for p in Path(args.images).glob("*.tif")}
        masks = {p.stem for p in Path(args.multiclass_masks).glob("*.png")}
        if images != masks:
            problems.append(f"imagens ({len(images)}) e mascaras multiclasse ({len(masks)}) nao casam 1:1")
        if set(tile_split) - images:
            problems.append("tiles do split sem imagem")
        bad = sd.check_mask_palette(sorted(Path(args.multiclass_masks).glob("*.png"))[:25])
        if bad:
            problems.append(f"cores fora da paleta em mascaras multiclasse: {bad}")
    if any(EXPERIMENTS[e].get("init") for e in selected) and not P1_CHECKPOINT.is_file():
        problems.append(f"checkpoint P1 ausente: {P1_CHECKPOINT}")

    free = shutil.disk_usage(BASE_DIR).free
    if free < 10 * 1024**3:
        problems.append(f"apenas {free / 1024**3:.1f} GiB livres (recomendado >= 10 GiB)")
    if problems:
        raise RuntimeError("Pre-voo falhou:\n  - " + "\n  - ".join(problems))
    return split_path, split_info


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Somente lista os comandos.")
    p.add_argument("--only", choices=("train", "test", "summary"), action="append")
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--run-id")
    p.add_argument("--run-root", default=str(BASE_DIR / "runs"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--experiments", nargs="+", default=["all"], metavar="EXPERIMENT")
    p.add_argument("--split", default=None)
    p.add_argument("--manifest", default=str(DEFAULTS["manifest"]))
    p.add_argument("--species-names", default=str(DEFAULTS["species_names"]))
    p.add_argument("--images", default=str(DEFAULTS["images"]))
    p.add_argument("--multiclass-masks", default=str(DEFAULTS["multiclass_masks"]))
    return p.parse_args()


def select_experiments(requested):
    if requested == ["all"]:
        return list(EXPERIMENTS)
    for name in requested:
        if name in DISABLED:
            raise ValueError(f"Experimento desativado: {name}: {DISABLED[name]}")
        if name not in EXPERIMENTS:
            raise ValueError(f"Experimento desconhecido: {name}. Opcoes: {list(EXPERIMENTS)}")
    return list(dict.fromkeys(requested))


def run_logged(command, log_file, cwd, label):
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.perf_counter()
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n[{label}] command={json.dumps(command)}\n[{label}] started_at={started_at}\n")
        log.flush()
        try:
            subprocess.run(command, cwd=cwd, check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            log.write(f"[{label}] failed code={exc.returncode}\n")
            raise RuntimeError(f"Etapa {label} falhou com codigo {exc.returncode}") from exc
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"[{label}] ended_at={dt.datetime.now(dt.timezone.utc).isoformat()}\n")
        log.write(f"[{label}] elapsed_s={time.perf_counter() - t0:.3f}\n")


def build_commands(args, name, run_id, run_root, split_path):
    exp = EXPERIMENTS[name]
    run_dir = run_root / run_id
    py = [sys.executable, "-u"]
    steps = {}
    if exp["kind"] == "efficientnet":
        train = py + [str(BASE_DIR / exp["script"]), "--manifest", args.manifest, "--split", str(split_path),
                      "--species-names", args.species_names, "--runs-dir", str(run_root), "--run-id", run_id]
        if exp["augment"]:
            aug_dir = run_dir / "train_augmented"
            steps["augment"] = py + [str(BASE_DIR / "augment_rare_species.py"), "--manifest", args.manifest,
                                     "--split", str(split_path), "--out-dir", str(aug_dir),
                                     # 300/classe: treino ~2x maior; o padrao (maior classe=1000) daria ~4,7x
                                     "--target", str(AUGMENT_TARGET)]
            train += ["--augmented-manifest", str(aug_dir / "train_augmented.csv")]
        if args.smoke:
            train.append("--smoke")
        steps["train"] = train
        # teste unico ja executado ao final do treino (test_metrics_global.json)
        steps["test"] = None
    else:
        train = py + [str(BASE_DIR / "run_legacy_multiclass_training.py"), "--script", str(BASE_DIR / exp["script"]),
                      "--images-dir", args.images, "--masks-dir", args.multiclass_masks,
                      "--run-root", str(run_root), "--run-id", run_id, "--split", str(split_path),
                      "--species-names", args.species_names]
        if exp["init"]:
            train += ["--init-checkpoint", exp["init"]]
        if args.smoke:
            train.append("--smoke")
        steps["train"] = train
        test = py + [str(BASE_DIR / "evaluate_multiclass_test.py"), "--images", args.images,
                     "--masks", args.multiclass_masks, "--split", str(split_path),
                     "--checkpoint", str(run_dir / "checkpoint_best.pth"), "--output", str(run_dir / "test_eval"),
                     "--normalize", exp["normalize"], "--species-names", args.species_names]
        if args.smoke:
            test += ["--max-tiles", "10"]
        steps["test"] = test
    return {"run_id": run_id, "run_dir": str(run_dir), "experiment": name, **steps}


def main():
    args = parse_args()
    selected = select_experiments(args.experiments)
    stages = set(args.only or ("train", "test", "summary"))
    if args.skip_test:
        stages.discard("test")
    split_path, split_info = preflight(args, selected) if stages & {"train", "test"} else (
        sd.resolve_split_path(args.split), None)
    run_id = args.run_id or ("classif_" + ("smoke_" if args.smoke else "")
                             + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S"))
    run_root = Path(args.run_root)
    plan = {name: build_commands(args, name, f"{run_id}_{name}", run_root, split_path) for name in selected}

    if args.dry_run:
        print(json.dumps({"run_id": run_id, "stages": sorted(stages), "disabled": DISABLED,
                          "commands": plan}, indent=2, ensure_ascii=False))
        return

    results = {}
    for name, item in plan.items():
        run_dir = Path(item["run_dir"])
        if "train" in stages:
            if run_dir.exists():
                raise FileExistsError(f"Run ja existe, nao sobrescrevo: {run_dir}")
            run_dir.mkdir(parents=True)
        elif not run_dir.is_dir():
            print(f"[ERRO] {name}: run inexistente para teste {run_dir}", file=sys.stderr)
            continue
        log_file = run_dir / "suite_log.txt"
        (run_dir / "suite_command.json").write_text(json.dumps({
            "suite": Path(__file__).name, "suite_run_id": run_id, "experiment": name,
            "description": EXPERIMENTS[name]["description"], "split": split_info,
            "commands": {k: v for k, v in item.items() if isinstance(v, list)},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        status = "ok"
        try:
            if "train" in stages:
                if item.get("augment"):
                    run_logged(item["augment"], log_file, BASE_DIR, "augment")
                run_logged(item["train"], log_file, BASE_DIR, "train")
            if "test" in stages and item["test"] is not None:
                if not (run_dir / "checkpoint_best.pth").is_file():
                    raise RuntimeError(f"checkpoint ausente {run_dir / 'checkpoint_best.pth'}")
                run_logged(item["test"], log_file, BASE_DIR, "test")
        except Exception as exc:
            status = f"falhou: {exc}"
            print(f"[ERRO] {name}: {exc}", file=sys.stderr)
            (run_dir / "failure.json").write_text(json.dumps({"experiment": name, "error": str(exc)},
                                                             ensure_ascii=False, indent=2), encoding="utf-8")
        results[name] = {"run_dir": str(run_dir), "status": status}

    if "summary" in stages:
        summary_dir = run_root / f"{run_id}_suite"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary = {"run_id": run_id, "experiments": results, "disabled": DISABLED,
                   "split_sha256": sha256(split_path),
                   "ended_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
        subprocess.run([sys.executable, str(BASE_DIR / "build_classification_index.py")], cwd=BASE_DIR, check=True)
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
