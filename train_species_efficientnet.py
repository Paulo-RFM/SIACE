"""Treino auditavel do classificador de especies (EfficientNet-B0).

Regras aplicadas (ver species_data.py):
  * particao herdada do tile pelo split espacial persistido (tiles_split.csv);
  * validacao e teste contem SOMENTE recortes anotados manualmente;
  * aumentados e pseudorrotulos apenas no treino, ligados a arvore original;
  * pseudorrotulos exigem proveniencia completa e especie nao exclusivamente auto-predita;
  * o teste e avaliado uma unica vez, com o checkpoint escolhido na validacao.

Saida: runs/<run_id>/ com config.json, curvas (PNG+CSV), metricas globais e por
classe de validacao e teste, matrizes de confusao, classification_report,
calibracao, predicoes por amostra e o manifesto efetivamente usado.

Exemplos:
  python train_species_efficientnet.py --audit-only
  python train_species_efficientnet.py --smoke
  python train_species_efficientnet.py                       # treino completo
  python train_species_efficientnet.py --variant annotated_plus_pseudo --pseudo-csv X.csv
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import platform
import random
import resource
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import species_data as sd

LAB_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = sd.PROJECT_ROOT / "species_dataset_v2" / "species_manifest.csv"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--augmented-manifest", default=None,
                   help="CSV de augment_rare_species.py (somente treino)")
    p.add_argument("--pseudo-csv", default=None)
    p.add_argument("--variant", choices=["annotated_only", "annotated_plus_pseudo"], default="annotated_only")
    p.add_argument("--split", default=None, help="tiles_split.csv; resolvido automaticamente se omitido")
    p.add_argument("--species-names", default=None, help="JSON {id: nome}")
    p.add_argument("--runs-dir", default=str(LAB_DIR / "runs"))
    p.add_argument("--run-id", default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--cosine-tmax", type=int, default=10)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-test-samples", type=int, default=10)
    p.add_argument("--audit-only", action="store_true", help="so valida dados e imprime inventario")
    p.add_argument("--smoke", action="store_true", help="1 epoca, poucos lotes; mede tempo")
    p.add_argument("--smoke-batches", type=int, default=20)
    p.add_argument("--binary-target", type=int, default=None,
                   help="experimento binario: classe <id> contra todas as demais (rotulo 0 = 'outras'). "
                        "Mesmos recortes, mesma divisao e mesmo protocolo do multiclasse.")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------

def load_dataset(args):
    split_path = sd.resolve_split_path(args.split)
    tile_split, split_info = sd.load_tile_split(split_path)
    print(f"[split] {split_path}\n        sha256={split_info['split_sha256']}\n"
          f"        tiles por particao={split_info['tiles_per_split']} | tiles em >1 particao=0")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise sd.DataAuditError(f"Manifesto ausente: {manifest_path}. Rode build_species_manifest.py.")
    frames = [pd.read_csv(manifest_path)]
    sources = {"manifest": str(manifest_path), "manifest_sha256": sd.sha256_file(manifest_path)}
    if args.augmented_manifest:
        aug_path = Path(args.augmented_manifest)
        aug = pd.read_csv(aug_path)
        frames.append(aug[aug["origin"].astype(str).str.lower().isin({"aumentado", "augmented"})])
        sources.update(augmented_manifest=str(aug_path), augmented_manifest_sha256=sd.sha256_file(aug_path))
    if args.variant == "annotated_plus_pseudo":
        if not args.pseudo_csv:
            raise sd.DataAuditError("--variant annotated_plus_pseudo exige --pseudo-csv.")
        pseudo = pd.read_csv(args.pseudo_csv)
        if "origin" not in pseudo.columns:
            raise sd.DataAuditError("CSV de pseudorrotulos sem coluna 'origin'.")
        frames.append(pseudo)
        sources.update(pseudo_csv=str(args.pseudo_csv), pseudo_csv_sha256=sd.sha256_file(args.pseudo_csv))
    elif args.pseudo_csv:
        raise sd.DataAuditError("--pseudo-csv informado com --variant annotated_only.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    df = sd.validate_manifest(df, tile_split)
    missing_files = [p for p in df["crop_path"] if not Path(p).is_file()]
    if missing_files:
        raise sd.DataAuditError(f"{len(missing_files)} recortes ausentes, ex.: {missing_files[:3]}")
    return df, split_info, sources


def audit_summary(df, min_test):
    counts = sd.counts_by_split(df)
    counts_origin = pd.crosstab([df["split"], df["origin"]], df["species"])
    leak = sd.leakage_report(df)
    elig = sd.test_eligibility(df, min_test)
    print("\n[auditoria] amostras por especie e particao (todas as origens):")
    print(counts.assign(total=counts.sum(axis=1)))
    print("\n[auditoria] por particao e origem:")
    print(counts_origin)
    print("\n[auditoria] razao maior/menor classe:",
          {s: sd.imbalance_ratio(counts[s]) for s in sd.VALID_SPLITS})
    print("[auditoria] vazamento:", json.dumps(leak))
    for s in elig["below_min"]:
        print(f"AVISO: especie {s} tem {elig['test_counts'][s]} amostras no teste (< {min_test}); "
              "metricas por classe dessa especie nao sao estatisticamente estaveis.")
    print(f"[auditoria] especies com >= {min_test} amostras no teste: "
          f"{elig['eligible_species']} ({elig['n_eligible']})")
    if not elig["meaningful_test"]:
        raise sd.DataAuditError(
            f"Nenhuma especie tem >= {min_test} amostras anotadas no teste; treino bloqueado.")
    return {"counts_species_by_split": {str(k): {c: int(v) for c, v in r.items()} for k, r in counts.iterrows()},
            "counts_by_split_origin_species": {f"{a}|{b}": {str(k): int(v) for k, v in r.items()}
                                               for (a, b), r in counts_origin.iterrows()},
            "imbalance_ratio": {s: sd.imbalance_ratio(counts[s]) for s in sd.VALID_SPLITS},
            "samples_per_split": df["split"].value_counts().to_dict(),
            "leakage": leak, "test_eligibility": elig}


# --------------------------------------------------------------------------
# Ambiente
# --------------------------------------------------------------------------

def environment_info(torch):
    import torchvision
    info = {"python": platform.python_version(), "pytorch": torch.__version__,
            "torchvision": torchvision.__version__, "host": socket.gethostname(),
            "os": platform.platform(), "cuda_available": torch.cuda.is_available()}
    try:
        import efficientnet_pytorch
        info["efficientnet_pytorch"] = efficientnet_pytorch.__version__
    except Exception as exc:  # pragma: no cover
        info["efficientnet_pytorch"] = f"indisponivel: {exc}"
    import sklearn
    info["scikit_learn"] = sklearn.__version__
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info.update(gpu=prop.name, vram_total_mb=round(prop.total_memory / 2**20, 2),
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    return info


def main(argv=None):
    args = parse_args(argv)
    df, split_info, sources = load_dataset(args)
    audit = audit_summary(df, args.min_test_samples)

    class_ids = sorted(int(s) for s in df["species"].unique())
    names_map, names_from_file = sd.load_species_names(args.species_names, class_ids)
    class_names = [names_map[i] for i in class_ids]
    if args.binary_target is not None:
        t = args.binary_target
        if t not in class_ids:
            raise sd.DataAuditError(f"--binary-target {t} nao existe no manifesto ({class_ids}).")
        df["species_original"] = df["species"]
        df["species"] = np.where(df["species"] == t, t, 0)
        class_ids, class_names = [0, t], ["outras", names_map[t]]
        print(f"[binario] {class_names[1]} (id {t}) contra as demais:")
        print(pd.crosstab(df["species"], df["split"]))
    if not names_from_file:
        print("AVISO: sem --species-names; usando 'especie_<id>'. Nomes cientificos pendentes.")
    id_to_idx = {sid: k for k, sid in enumerate(class_ids)}
    df["label_index"] = df["species"].map(id_to_idx)

    if args.audit_only:
        print("\n--audit-only: nenhum treinamento executado.")
        return audit

    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from efficientnet_pytorch import EfficientNet
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    import species_metrics as sm

    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: adaptive_avg_pool2d (EfficientNet) nao tem backward determinista em CUDA.
    torch.use_deterministic_algorithms(True, warn_only=True)

    def seed_worker(worker_id):
        ws = torch.initial_seed() % 2**32
        np.random.seed(ws); random.seed(ws)

    run_id = args.run_id or ("species_" + ("smoke_" if args.smoke else "") + args.variant
                             + (f"_binaria_{args.binary_target}" if args.binary_target is not None else "") + "_"
                             + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = Path(args.runs_dir) / run_id
    # Diretorio pode ter sido criado pela suite (suite_log.txt, suite_command.json,
    # train_augmented/); recusa apenas se ja houver artefatos de treino.
    if (run_dir / "config.json").exists() or (run_dir / "checkpoint_best.pth").exists():
        raise FileExistsError(f"Run ja existe, nao sobrescrevo: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"run_id={run_id} device={device}")
    df.to_csv(run_dir / "species_run_manifest.csv", index=False)
    test_paths = sorted(df.loc[df.split == "test", "crop_path"])
    (run_dir / "test_list.txt").write_text("\n".join(test_paths) + "\n", encoding="utf-8")

    class CropDataset(Dataset):
        def __init__(self, frame, transform):
            self.paths = frame["crop_path"].tolist()
            self.labels = frame["label_index"].astype(int).tolist()
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            return self.transform(img), self.labels[i]

    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = T.Compose([
        T.Resize((args.img_size, args.img_size)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(30), T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        T.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)), T.ToTensor(), norm])
    eval_tf = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor(), norm])

    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df = df[df.split == "val"].reset_index(drop=True)
    test_df = df[df.split == "test"].reset_index(drop=True)

    freq = train_df["label_index"].value_counts()
    weights = train_df["label_index"].map(lambda k: 1.0 / freq[k]).to_numpy()
    g = torch.Generator(); g.manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights),
                                    replacement=True, generator=g)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=seed_worker,
                     pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(CropDataset(train_df, train_tf), sampler=sampler, generator=g, **loader_kw)
    val_loader = DataLoader(CropDataset(val_df, eval_tf), shuffle=False, **loader_kw)
    test_loader = DataLoader(CropDataset(test_df, eval_tf), shuffle=False, **loader_kw)

    num_classes = len(class_ids)
    model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_tmax)
    epochs = 1 if args.smoke else args.epochs
    labels_all = list(range(num_classes))

    config = {
        "run_id": run_id, "script": Path(__file__).name, "command": [sys.executable] + sys.argv,
        "variant": args.variant, "smoke": args.smoke, "binary_target": args.binary_target,
        "seed": seed, "random_seed": seed, "numpy_seed": seed, "torch_manual_seed": seed, "python_seed": seed,
        "worker_init_fn": "seed_worker: torch.initial_seed() -> numpy/random por worker",
        "deterministic": "cudnn.deterministic=True, benchmark=False, use_deterministic_algorithms(warn_only=True)",
        "task": "classificacao de especies por recorte de copa",
        "architecture": "EfficientNet-B0 (classificador)", "backbone": "efficientnet-b0",
        "pretrained": "ImageNet (efficientnet_pytorch.from_pretrained)",
        "input_resolution": [args.img_size, args.img_size], "batch_size": args.batch_size,
        "learning_rate": args.lr, "weight_decay": args.weight_decay, "optimizer": "AdamW",
        "scheduler": f"CosineAnnealingLR(T_max={args.cosine_tmax}) step por epoca",
        "loss": "CrossEntropyLoss", "sampler": "WeightedRandomSampler(1/freq da classe no treino)",
        "train_augmentation": str(train_tf), "epochs_max": epochs, "patience": args.patience,
        "selection_criterion": "max val_f1_macro", "total_parameters": total_params,
        "trainable_parameters": trainable_params, "class_ids": class_ids, "class_names": class_names,
        "class_names_from_file": names_from_file, "split": split_info, "data_sources": sources,
        "test_list_sha256": sd.sha256_text_lines(test_paths), "audit": audit,
        "mask_provenance": sorted(df.get("mask_provenance", pd.Series(["desconhecida"])).dropna().unique().tolist()),
        "segmentation_checkpoint": None if df.get("segmentation_checkpoint", pd.Series([None])).isna().all()
        else sorted(df["segmentation_checkpoint"].dropna().unique().tolist()),
        "segmentation_checkpoint_sha256": None,
        "watershed_parameters": None,
        "segmentation_note": ("Recortes de treino/val/teste vem de poligonos manuais (LabelMe); "
                              "nao dependem de mascara predita nem de watershed."),
        "environment": environment_info(torch),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    sm.write_json(run_dir / "config.json", config)

    def run_epoch(loader, train: bool, max_batches=None):
        model.train(train)
        losses, preds, trues, probs = [], [], [], []
        with torch.set_grad_enabled(train):
            for b, (x, y) in enumerate(loader):
                if max_batches is not None and b >= max_batches:
                    break
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                if train:
                    optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                if train:
                    loss.backward(); optimizer.step()
                losses.append(loss.item() * len(y))
                preds.extend(out.argmax(1).cpu().tolist()); trues.extend(y.cpu().tolist())
                if not train:
                    probs.append(torch.softmax(out.float(), 1).cpu().numpy())
        n = max(len(trues), 1)
        return (sum(losses) / n, np.array(trues), np.array(preds),
                np.concatenate(probs) if probs else None)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    history, best_f1, best_epoch, es = [], -1.0, 0, 0
    t_start = time.time()
    max_b = args.smoke_batches if args.smoke else None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_y, tr_p, _ = run_epoch(train_loader, True, max_b)
        va_loss, va_y, va_p, _ = run_epoch(val_loader, False, max_b)
        scheduler.step()
        _, _, f_cls, _ = precision_recall_fscore_support(va_y, va_p, labels=labels_all, zero_division=0)
        vp, vr, vf, _ = precision_recall_fscore_support(va_y, va_p, labels=labels_all, average="macro",
                                                         zero_division=0)
        row = {"epoch": epoch, "lr": lr_now,
               "train_loss": tr_loss, "train_accuracy": accuracy_score(tr_y, tr_p),
               "train_f1_macro": f1_score(tr_y, tr_p, labels=labels_all, average="macro", zero_division=0),
               "val_loss": va_loss, "val_accuracy": accuracy_score(va_y, va_p),
               "val_precision_macro": vp, "val_recall_macro": vr, "val_f1_macro": vf,
               "epoch_time_s": time.time() - t0}
        row.update({f"val_f1_class_{class_names[k]}": float(f_cls[k]) for k in labels_all})
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "learning_curve_species.csv", index=False)
        log(f"epoch {epoch}/{epochs} lr={lr_now:.2e} train_loss={tr_loss:.4f} train_f1={row['train_f1_macro']:.4f} "
            f"val_loss={va_loss:.4f} val_f1={vf:.4f} val_acc={row['val_accuracy']:.4f} "
            f"({row['epoch_time_s']:.1f}s)")
        if vf > best_f1:
            best_f1, best_epoch, es = vf, epoch, 0
            torch.save(model.state_dict(), run_dir / "checkpoint_best.pth")
            log(f"  -> novo melhor val_f1_macro={vf:.4f}")
        else:
            es += 1
            if es >= args.patience:
                log(f"Early stopping apos {es} epocas sem melhora.")
                break
    torch.save(model.state_dict(), run_dir / "checkpoint_last.pth")
    train_time = time.time() - t_start
    hist_df = pd.DataFrame(history)
    sm.save_training_curves(hist_df, run_dir, class_names)

    # ---------------- Avaliacao final: validacao e teste (uma unica vez) ----------------
    model.load_state_dict(torch.load(run_dir / "checkpoint_best.pth", map_location=device, weights_only=True))
    results = {}
    for name, loader, frame in (("val", val_loader, val_df), ("test", test_loader, test_df)):
        t0 = time.time()
        loss, y, p, pr = run_epoch(loader, False, max_b)
        infer_s = time.time() - t0
        g_m = sm.global_metrics(y, p, labels_all)
        g_m.update(loss=loss, inference_total_s=infer_s, split=name,
                   checkpoint="checkpoint_best.pth", best_epoch=best_epoch)
        per_cls = sm.per_class_metrics(y, p, pr, labels_all, class_names, class_ids)
        per_cls.to_csv(run_dir / f"{name}_metrics_by_class.csv", index=False)
        pd.DataFrame([g_m]).to_csv(run_dir / f"{name}_metrics_global.csv", index=False)
        sm.write_json(run_dir / f"{name}_metrics_global.json", g_m)
        (run_dir / f"{name}_classification_report.txt").write_text(
            sm.text_report(y, p, labels_all, class_names), encoding="utf-8")
        sm.save_confusion(y, p, labels_all, class_names, run_dir, name,
                          f"EfficientNet-B0 - {name}")
        conf = sm.save_confidence_outputs(y, pr, class_names, run_dir, name)
        g_m["ece"] = conf["ece"]
        sm.write_json(run_dir / f"{name}_metrics_global.json", g_m)
        cols = ["crop_path", "tile", "species", "instance_uid"] + (["species_original"] if "species_original" in frame else [])
        pred_df = frame.iloc[:len(y)][cols].copy()
        pred_df["true_index"] = y; pred_df["pred_index"] = p
        pred_df["pred_species"] = [class_ids[i] for i in p]
        for k, n in enumerate(class_names):
            pred_df[f"prob_{n}"] = pr[:, k]
        pred_df.to_csv(run_dir / f"{name}_predictions.csv", index=False)
        results[name] = g_m
        log(f"[{name}] " + json.dumps({k: g_m[k] for k in ("accuracy", "balanced_accuracy", "f1_macro",
                                                         "f1_weighted", "number_of_samples")}))

    peak_gpu = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
    config.update({
        "ended_at": datetime.now(timezone.utc).isoformat(), "effective_epoch": best_epoch,
        "epochs_completed": len(history), "best_val_f1_macro": best_f1,
        "train_duration_s": train_time, "mean_epoch_duration_s": float(hist_df["epoch_time_s"].mean()),
        "peak_gpu_memory_allocated_mb": peak_gpu,
        "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "checkpoint_best_sha256": sd.sha256_file(run_dir / "checkpoint_best.pth"),
        "results": results,
        "pseudo_threshold_note": sm.PSEUDO_THRESHOLD_NOTE,
    })
    if args.smoke:
        n_batches = int(np.ceil(len(train_df) / args.batch_size)) + int(np.ceil(len(val_df) / args.batch_size))
        per_batch = hist_df["epoch_time_s"].iloc[0] / max(1, 2 * args.smoke_batches)
        config["smoke_estimate"] = {
            "batches_per_full_epoch": n_batches,
            "seconds_per_batch_observed": per_batch,
            "estimated_full_epoch_s": per_batch * n_batches,
            "note": "estimativa grosseira: lotes de validacao sao mais rapidos que os de treino",
        }
    sm.write_json(run_dir / "config.json", config)
    log(f"Run salvo em {run_dir}")
    return config


if __name__ == "__main__":
    try:
        main()
    except sd.DataAuditError as exc:
        print(f"\nBLOQUEADO PELA AUDITORIA: {exc}", file=sys.stderr)
        sys.exit(2)
