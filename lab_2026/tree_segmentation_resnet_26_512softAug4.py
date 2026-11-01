import os
import random
import argparse
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import DiceLoss

from torch.cuda.amp import autocast, GradScaler
from datetime import timedelta
import time
import uuid
import gc
from pathlib import Path
from unet_spatial_protocol import (
    filter_black_tiles,
    load_split,
    paired_tiles,
    frozen_threshold,
    valid_pixel_mask,
    runtime_metadata,
    freeze_validation_threshold,
)

# ==============================
# CONFIGURAÇÕES (sem almento de dados)
# ==============================

parser = argparse.ArgumentParser()
parser.add_argument("--images-dir", default="/home/paulomaciel/proj/imagens/")
parser.add_argument("--masks-dir", default="/home/paulomaciel/proj/masks/")
parser.add_argument("--run-root", default="/home/paulomaciel/proj/lab_2026/runs")
parser.add_argument("--run-id")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--smoke", action="store_true", help="Executa ate uma epoca para validacao.")
args = parser.parse_args()

base_image_tiles_dir = args.images_dir
base_mask_tiles_dir = args.masks_dir
run_id = args.run_id or uuid.uuid4().hex[:12]
run_dir = os.path.join(args.run_root, run_id)
os.makedirs(run_dir, exist_ok=True)
checkpoint_path = os.path.join(run_dir, "checkpoint_best.pth")
last_checkpoint_path = os.path.join(run_dir, "checkpoint_last.pth")
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

img_size = (512,512)
batch_size = 6
epochs = 1 if args.smoke else min(args.epochs, 40)
patience = 5

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.use_deterministic_algorithms(True)
except (RuntimeError, TypeError):
    print("Aviso: esta versao do PyTorch nao habilita todos os algoritmos deterministas.")

def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# UTILS
# ==============================

def list_all_files(directory, extension='.png'):
    result = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(extension):
                result.append(os.path.join(root, f))
    return sorted(result)

def rgb_to_binary(mask_rgb):
    """
    Converte qualquer pixel que não seja preto (0,0,0) em 1 (árvore).
    """
    mask = np.any(mask_rgb != [0, 0, 0], axis=-1)
    return mask.astype(np.uint8)

# ==============================
# DATASET
# ==============================

class TreeBinaryDataset(Dataset):
    def __init__(self, image_files, mask_files, img_size, train=True):
        self.image_files = image_files
        self.mask_files = mask_files
        self.img_size = img_size
        self.train = train

        # AUGMENTAÇÃO
        self.crop = transforms.RandomResizedCrop(
            img_size,
            scale=(0.85, 1.0)
        )

        self.color_jitter = transforms.ColorJitter(
            brightness=0.05,
            contrast=0.05,
            saturation=0.05
        )

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        mask = Image.open(self.mask_files[idx]).convert("RGB")

        if self.train:
            seed = np.random.randint(0, 100000)

            # =========================
            # CROP SINCRONIZADO
            # =========================
            random.seed(seed)
            img = self.crop(img)

            random.seed(seed)
            mask = self.crop(mask)

            # =========================
            # FLIPS
            # =========================
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # COLOR JITTER (só imagem)
            # =========================
            img = self.color_jitter(img)

        else:
            img = img.resize(self.img_size, Image.BILINEAR)
            mask = mask.resize(self.img_size, Image.NEAREST)

        # =========================
        # MÁSCARA BINÁRIA
        # =========================
        mask_np = rgb_to_binary(np.array(mask))
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
        valid = torch.tensor(valid_pixel_mask(img), dtype=torch.bool)

        img = self.to_tensor(img)
        img = self.normalize(img)

        return img, mask_tensor, valid
# ==============================
# CARREGAR ARQUIVOS
# ==============================

SPLIT_DIR = os.environ.get(
    "UNET_SPATIAL_SPLIT_DIR",
    "/home/paulomaciel/proj/unet_spatial_split",
)
THRESHOLD_PATH = os.path.join(SPLIT_DIR, "frozen_threshold.json")

pairs = paired_tiles(base_image_tiles_dir, base_mask_tiles_dir)
pairs, excluded_black = filter_black_tiles(pairs)
image_files = [image for image, _ in pairs]
split_by_name, split_mapping, split_manifest = load_split(SPLIT_DIR, image_files)
image_train = split_by_name["train"]
image_val = split_by_name["val"]
mask_by_image = {os.path.splitext(os.path.basename(image))[0]: mask for image, mask in pairs}
mask_train = [mask_by_image[os.path.splitext(os.path.basename(image))[0]] for image in image_train]
mask_val = [mask_by_image[os.path.splitext(os.path.basename(image))[0]] for image in image_val]

def positive_fraction(image_paths, mask_paths):
    foreground = 0
    valid_count = 0
    for image_path, mask_path in zip(image_paths, mask_paths):
        mask = np.asarray(Image.open(mask_path).convert("RGB"))
        label = np.any(mask != 0, axis=2)
        valid = valid_pixel_mask(Image.open(image_path).convert("RGB"))
        foreground += int(label[valid].sum())
        valid_count += int(valid.sum())
    return foreground / max(valid_count, 1)

train_positive_fraction = positive_fraction(image_train, mask_train)
val_positive_fraction = positive_fraction(image_val, mask_val)
print("Imagens:", len(pairs))
print("Pretos integralmente excluidos:", len(excluded_black))

train_dataset = TreeBinaryDataset(image_train, mask_train, img_size, train=True)
val_dataset   = TreeBinaryDataset(image_val  , mask_val  , img_size, train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                          pin_memory=True, persistent_workers=True, drop_last=False,
                          worker_init_fn=seed_worker)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                          pin_memory=True, persistent_workers=True, drop_last=False,
                          worker_init_fn=seed_worker)

# ==============================
# MODELO
# ==============================

model = Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1   # <- binário
).to(device)

# LOSS BINÁRIA
bce = nn.BCEWithLogitsLoss()
dice = DiceLoss(mode="binary", from_logits=True)

def loss_fn(pred, target, valid):
    valid = valid.unsqueeze(1)
    bce_per_pixel = nn.functional.binary_cross_entropy_with_logits(
        pred, target, reduction="none"
    )
    bce_value = (bce_per_pixel * valid).sum() / valid.sum().clamp_min(1)
    masked_pred = pred * valid
    masked_target = target * valid
    return 0.5 * bce_value + 0.5 * dice(masked_pred, masked_target)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

scaler = GradScaler()

# ==============================
# MÉTRICAS
# ==============================

def bin_metrics(pred, target):
    """
    pred: logits
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    TP = (pred * target).sum().item()
    FP = (pred * (1 - target)).sum().item()
    FN = ((1 - pred) * target).sum().item()

    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall)
    iou = TP / max(TP + FP + FN, 1)

    acc = (pred == target).float().mean().item()

    return acc, precision, recall, f1, iou

# ==============================
# TREINAMENTO
# ==============================

best_iou = -1
no_improve = 0
history = []

run_started = time.perf_counter()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=3,factor=0.5)

for epoch in range(epochs):

    # =========================
    # TREINO
    # =========================
    model.train()
    train_loss = 0

    epoch_started = time.perf_counter()
    for img, mask, valid in train_loader:
        img, mask = img.to(device), mask.to(device)
        valid = valid.to(device)

        optimizer.zero_grad()

        with autocast(enabled=(device.type == "cuda")):
            pred = model(img)
            loss = loss_fn(pred, mask, valid)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # =========================
    # VALIDAÇÃO; 0,5 é apenas métrico por época.
    # A curva de limiar é calculada uma vez ao final.
    # =========================
    model.eval()
    val_loss = 0
    val_started = time.perf_counter()

    TP_total = FP_total = FN_total = TN_total = 0
    val_loss_temp = 0

    with torch.no_grad():
        for img, mask, valid in val_loader:
            img, mask, valid = img.to(device), mask.to(device), valid.to(device)
            with autocast(enabled=(device.type == "cuda")):
                pred = model(img)
                loss = loss_fn(pred, mask, valid)
            val_loss_temp += loss.item()

            pred_bin = (torch.sigmoid(pred) >= 0.5).float()
            valid_expanded = valid.unsqueeze(1).bool()
            pred_valid = pred_bin[valid_expanded].view(-1)
            mask_valid = mask[valid_expanded].view(-1)
            TP_total += ((pred_valid == 1) & (mask_valid == 1)).sum().item()
            FP_total += ((pred_valid == 1) & (mask_valid == 0)).sum().item()
            FN_total += ((pred_valid == 0) & (mask_valid == 1)).sum().item()
            TN_total += ((pred_valid == 0) & (mask_valid == 0)).sum().item()

    precision = TP_total / max(TP_total + FP_total, 1)
    recall = TP_total / max(TP_total + FN_total, 1)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    iou = TP_total / max(TP_total + FP_total + FN_total, 1)
    acc = (TP_total + TN_total) / max(TP_total + TN_total + FP_total + FN_total, 1)
    best_metrics = {
        "precision": precision, "recall": recall, "f1": f1, "acc": acc,
        "tp": TP_total, "fp": FP_total, "fn": FN_total, "tn": TN_total,
        "val_loss": val_loss_temp / len(val_loader), "threshold": 0.5, "iou": iou,
    }

    # =========================
    # RESULTADOS DA ÉPOCA
    # =========================
    val_loss = best_metrics["val_loss"]
    precision = best_metrics["precision"]
    recall = best_metrics["recall"]
    f1 = best_metrics["f1"]
    acc = best_metrics["acc"]
    iou = best_metrics["iou"]


    tempo_val_s = time.perf_counter() - val_started
    scheduler.step(iou)
    current_lr = optimizer.param_groups[0]['lr']

    # =========================
    # LOG
    # =========================
    print(f"\n=== Época {epoch+1}/{epochs} ===")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")

    print("\n--- MÉTRICAS ---")
    print(f"Acc:  {acc:.4f}")
    print(f"Prec: {precision:.4f}")
    print(f"Rec:  {recall:.4f}")
    print(f"F1:   {f1:.4f}")
    print(f"IoU:  {iou:.4f}")
    print("Threshold de validacao atual: 0.50")

    print("\n--- MATRIZ DE CONFUSÃO ---")
    print(f"TP: {best_metrics['tp']}")
    print(f"FP: {best_metrics['fp']}")
    print(f"FN: {best_metrics['fn']}")
    print(f"TN: {best_metrics['tn']}")
    print(f"LR atual: {current_lr:.6f}")
    # =========================
    # SALVAR HISTÓRICO
    # =========================
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "threshold": 0.5,
        "tp": best_metrics["tp"],
        "fp": best_metrics["fp"],
        "fn": best_metrics["fn"],
        "tn": best_metrics["tn"],
        "model": "Unet_resnet34",
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "img_size": 512,
        "batch_size": batch_size,
        "lr": 1e-4
        ,
        "split": "val",
        "lr_efetiva": current_lr,
        "gpu_mem_alloc_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2)
        if torch.cuda.is_available() else 0,
        "gpu_mem_res_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2)
        if torch.cuda.is_available() else 0,
        "seed": 42,
        "run_id": run_id,
        "tempo_epoca_s": round(time.perf_counter() - epoch_started, 3),
        "tempo_val_s": round(tempo_val_s, 3),
        "fracao_pixels_positivos": val_positive_fraction,
    })

    # =========================
    # SALVAR MODELOS
    # =========================
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), last_checkpoint_path)

    df = pd.DataFrame(history)
    metrics_path = os.path.join(
        os.path.dirname(checkpoint_path),
        f"unet_metrics_512_softAug2_{run_id}.csv",
    )
    fixed_metrics_path = os.path.join(run_dir, "metrics.csv")
    df.to_csv(metrics_path, index=False)
    df.to_csv(fixed_metrics_path, index=False)
    print("CSV atualizado com sucesso")
    print(f"CSV do run salvo em: {metrics_path}")

    if iou > best_iou:
        best_iou = iou
        no_improve = 0

        torch.save(model.state_dict(), checkpoint_path)
        print(f"\n✔ Modelo salvo! (melhor IoU: {best_iou:.4f})")

    else:
        no_improve += 1
        print(f"Sem melhora: {no_improve}/{patience}")

        if no_improve >= patience:
            print("\n🛑 Early Stopping ativado")
            break

torch.save(model.state_dict(), last_checkpoint_path)
if os.path.isfile(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
model.eval()
validation_probabilities = []
validation_targets = []
validation_valid_pixels = []
with torch.no_grad():
    for img, mask, valid in val_loader:
        img = img.to(device)
        logits = model(img)
        validation_probabilities.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
        validation_targets.append(mask.numpy().reshape(-1))
        validation_valid_pixels.append(valid.numpy().reshape(-1))

threshold_values = np.arange(0.1, 0.91, 0.05)
frozen_threshold_value = freeze_validation_threshold(
    np.concatenate(validation_probabilities),
    np.concatenate(validation_targets),
    np.concatenate(validation_valid_pixels),
    run_dir,
    thresholds=threshold_values,
)
print(f"Limiar congelado pela validacao: {frozen_threshold_value:.2f}")
print(f"Curva salva em: {run_dir}/threshold_iou_curve.csv")
gc.collect()

metadata = runtime_metadata(
    {
        "seed": 42,
        "img_size": img_size,
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau(mode=max, patience=3, factor=0.5)",
        "loss": "0.5 BCEWithLogits + 0.5 Dice",
        "epochs_max": epochs,
        "patience": patience,
        "weight_decay": 1e-4,
        "model": "Unet_resnet34",
    },
    os.path.dirname(checkpoint_path),
    SPLIT_DIR,
    split_by_name["test"],
    run_id,
)

from unet_spatial_protocol import hash_files
metadata["split_hash"] = hash_files([os.path.join(SPLIT_DIR, "tiles_split.csv")])
metadata["split_manifest_hash"] = hash_files([os.path.join(SPLIT_DIR, "tiles_by_split.json")])

positive_fraction = {}
for split_name, image_paths in split_by_name.items():
    mask_paths = [mask_by_image[os.path.splitext(os.path.basename(path))[0]] for path in image_paths]
    positive_pixels = 0
    valid_pixels = 0
    for image_path, mask_path in zip(image_paths, mask_paths):
        mask = np.array(Image.open(mask_path).convert("RGB"))
        foreground = np.any(mask != 0, axis=2)
        valid = valid_pixel_mask(Image.open(image_path).convert("RGB"))
        positive_pixels += int(foreground[valid].sum())
        valid_pixels += int(valid.sum())
    positive_fraction[split_name] = positive_pixels / max(valid_pixels, 1)
metadata["positive_fraction_by_split"] = positive_fraction

Path(run_dir, "config.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
Path(run_dir, "split_hash.txt").write_text(
    metadata["split_hash"] + "\n", encoding="utf-8"
)
