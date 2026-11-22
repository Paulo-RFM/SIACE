#!/usr/bin/env python3
"""Adaptador dos scripts multiclasse (resnet2.py, resnet3.py) ao protocolo espacial.

Mesmo padrao de run_legacy_binary_training.py:
  * train_test_split -> particoes train/val do tiles_split.csv (test excluido);
  * torch.save do state_dict -> <run_dir>/checkpoint_best.pth (nunca sobrescreve .pth historicos);
  * DataFrame.to_csv do historico -> <run_dir>/metrics.csv;
  * checkpoint_path aponta para o run novo, entao o script nao retoma pesos antigos;
  * --init-checkpoint opcional: inicializa a U-Net com pesos de um run binario
    (ex.: runs/p1_20261101_140519/checkpoint_best.pth), exceto a cabeca de segmentacao;
  * grava config.json do run com proveniencia e ambiente.
O teste NAO e avaliado aqui (ver evaluate_multiclass_test.py).
"""
import argparse
import json
import platform
import random
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import torch.utils.data

import species_data as sd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--script", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--masks-dir", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--split", default=None)
    p.add_argument("--init-checkpoint", default=None)
    p.add_argument("--species-names", default=str(Path(__file__).resolve().parent / "species_names.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-batches", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    script = Path(args.script).resolve()
    run_dir = Path(args.run_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint_best.pth"
    if checkpoint_path.exists():
        raise FileExistsError(f"Run ja tem checkpoint, nao sobrescrevo: {checkpoint_path}")

    split_path = sd.resolve_split_path(args.split)
    tile_split, split_info = sd.load_tile_split(split_path)
    names_map, _ = sd.load_species_names(args.species_names, range(1, 7))

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    split_report = {}

    def spatial_train_test_split(first, second=None, *positional, **kwargs):
        first_items = list(first)
        second_items = None if second is None else list(second)
        idx = {"train": [], "val": [], "test": [], "fora_do_split": []}
        for i, item in enumerate(first_items):
            idx[tile_split.get(Path(str(item)).stem, "fora_do_split")].append(i)
        if second_items is not None:
            pairs_ok = all(Path(str(a)).stem == Path(str(b)).stem for a, b in zip(first_items, second_items))
            if not pairs_ok or len(first_items) != len(second_items):
                raise sd.DataAuditError("Imagens e mascaras nao pareadas pelo nome do tile.")
        if idx["fora_do_split"]:
            raise sd.DataAuditError(f"{len(idx['fora_do_split'])} tiles fora do split espacial.")
        if not idx["train"] or not idx["val"]:
            raise sd.DataAuditError("Particao train ou val vazia.")
        split_report.update({k: len(v) for k, v in idx.items()})
        pick = lambda items, ids: [items[i] for i in ids]
        if second_items is None:
            return pick(first_items, idx["train"]), pick(first_items, idx["val"])
        return (pick(first_items, idx["train"]), pick(first_items, idx["val"]),
                pick(second_items, idx["train"]), pick(second_items, idx["val"]))

    original_tts = sklearn.model_selection.train_test_split
    sklearn.model_selection.train_test_split = spatial_train_test_split

    original_save = torch.save
    saves = {"n": 0}

    def save_to_run_dir(obj, destination, *a, **kw):
        if str(destination).endswith(".pth") and isinstance(obj, dict):
            saves["n"] += 1
            return original_save(obj, checkpoint_path, *a, **kw)
        return original_save(obj, destination, *a, **kw)

    torch.save = save_to_run_dir
    original_to_csv = pd.DataFrame.to_csv

    def metrics_to_run_dir(self, destination=None, *a, **kw):
        return original_to_csv(self, run_dir / "metrics.csv", *a, **kw)

    pd.DataFrame.to_csv = metrics_to_run_dir

    original_loader = torch.utils.data.DataLoader
    if args.smoke:
        class SmokeLoader(original_loader):
            def __iter__(self):
                for b, batch in enumerate(super().__iter__()):
                    if b >= args.smoke_batches:
                        break
                    yield batch

            def __len__(self):
                return min(args.smoke_batches, super().__len__())
        torch.utils.data.DataLoader = SmokeLoader

    import segmentation_models_pytorch as smp
    original_unet = smp.Unet
    init_report = {}
    if args.init_checkpoint:
        init_state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)

        def unet_with_init(*a, **kw):
            model = original_unet(*a, **kw)
            own = model.state_dict()
            usable = {k: v for k, v in init_state.items()
                      if k in own and own[k].shape == v.shape and not k.startswith("segmentation_head.")}
            missing, unexpected = model.load_state_dict(usable, strict=False)
            init_report.update(init_checkpoint=str(args.init_checkpoint),
                               init_checkpoint_sha256=sd.sha256_file(args.init_checkpoint),
                               loaded_tensors=len(usable), not_loaded=sorted(missing))
            print(f"[init] {len(usable)} tensores carregados de {args.init_checkpoint}; "
                  f"nao carregados: {sorted(missing)}")
            return model
        smp.Unet = unet_with_init

    source = script.read_text(encoding="utf-8")
    for var, value in (("base_image_tiles_dir", args.images_dir), ("base_mask_tiles_dir", args.masks_dir),
                       ("checkpoint_path", str(checkpoint_path))):
        source, n = re.subn(rf"^{var}\s*=.*$", f"{var} = {value!r}", source, count=1, flags=re.MULTILINE)
        if n != 1:
            raise RuntimeError(f"{script.name}: variavel {var} nao encontrada para redirecionar.")
    if args.smoke:
        source = re.sub(r"^epochs\s*=.*$", "epochs = 1", source, count=1, flags=re.MULTILINE)
    normalize = "imagenet" if "transforms.Normalize" in source else "none"

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0, str(script.parent))
    try:
        namespace = {"__name__": "__main__", "__file__": str(script)}
        exec(compile(source, str(script), "exec"), namespace, namespace)
    finally:
        torch.save = original_save
        pd.DataFrame.to_csv = original_to_csv
        sklearn.model_selection.train_test_split = original_tts
        torch.utils.data.DataLoader = original_loader
        smp.Unet = original_unet

    if not checkpoint_path.is_file():
        raise RuntimeError(f"Script multiclasse nao salvou checkpoint: {script.name}")
    history = pd.read_csv(run_dir / "metrics.csv") if (run_dir / "metrics.csv").is_file() else pd.DataFrame()
    selection = namespace.get("selection_metric", "desconhecido")
    best_epoch = None
    if len(history) and selection in history.columns:
        best_epoch = int(history.loc[history[selection].idxmax() if selection != "val_loss"
                                     else history[selection].idxmin(), "epoch"])
    model = namespace.get("model")
    env = {"python": platform.python_version(), "pytorch": torch.__version__,
           "segmentation_models_pytorch": smp.__version__, "host": socket.gethostname(), "os": platform.platform()}
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        env.update(gpu=prop.name, vram_total_mb=round(prop.total_memory / 2**20, 2), cuda=torch.version.cuda)
    config = {
        "run_id": args.run_id, "script": script.name, "adapter": Path(__file__).name,
        "task": "classificacao multiclasse por pixel (segmentacao de especies)",
        "architecture": "U-Net", "encoder": "resnet34",
        "num_classes": int(namespace.get("num_classes", 7)),
        "class_names": ["fundo"] + [names_map[i] for i in range(1, 7)],
        "color_to_class": {str(k): v for k, v in namespace.get("color_to_class", {}).items()},
        "normalize": normalize, "img_size": list(namespace.get("img_size", (512, 512))),
        "batch_size": namespace.get("batch_size"), "epochs_max": namespace.get("epochs"),
        "patience": namespace.get("patience"), "selection_metric": selection,
        "loss": namespace.get("loss_description", "ver script"),
        "optimizer": type(namespace["optimizer"]).__name__ if "optimizer" in namespace else None,
        "class_weights": namespace.get("weights"),
        "seed": args.seed, "smoke": args.smoke,
        "split": {**split_info, "tiles_used": split_report},
        "masks_dir": args.masks_dir, "images_dir": args.images_dir,
        "init": init_report or None,
        "total_parameters": sum(p.numel() for p in model.parameters()) if model is not None else None,
        "epochs_completed": int(len(history)), "effective_epoch": best_epoch,
        "started_at": started_at, "ended_at": datetime.now(timezone.utc).isoformat(),
        "train_duration_s": time.perf_counter() - started,
        "mean_epoch_duration_s": float(history["epoch_time_s"].mean()) if "epoch_time_s" in history else None,
        "peak_gpu_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        "checkpoint_best_sha256": sd.sha256_file(checkpoint_path), "checkpoint_saves": saves["n"],
        "environment": env,
        "note": "Metricas por epoca sao de validacao; teste em test_eval/ (evaluate_multiclass_test.py).",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    print(f"Modelo multiclasse treinado com split espacial: {script.name} -> {run_dir}")


if __name__ == "__main__":
    main()
