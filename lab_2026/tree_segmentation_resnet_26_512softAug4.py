import os
import random
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

# ==============================
# CONFIGURAÇÕES (sem almento de dados)
# ==============================

base_image_tiles_dir = "/home/paulomaciel/proj/imagens/"
base_mask_tiles_dir  = "/home/paulomaciel/proj/masks/"
checkpoint_path      = "/home/paulomaciel/proj/lab_2026/unet_binary_tree_detector512_softAug_run2.pth"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

img_size = (512,512)
batch_size = 6
epochs = 50
patience = 8

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
            scale=(0.75, 1.0)
        )

        self.color_jitter = transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        )

        self.blur = transforms.GaussianBlur(kernel_size=3)

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

            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

            # =========================
            # ROTAÇÃO (CRÍTICO)
            # =========================
            if random.random() < 0.5:
                angle = random.choice([90, 180, 270])
                img = img.rotate(angle)
                mask = mask.rotate(angle)

            # =========================
            # COLOR JITTER (só imagem)
            # =========================
            img = self.color_jitter(img)

            # =========================
            # BLUR (opcional)
            # =========================
            if random.random() < 0.3:
                img = self.blur(img)

        else:
            img = img.resize(self.img_size, Image.BILINEAR)
            mask = mask.resize(self.img_size, Image.NEAREST)

        # =========================
        # MÁSCARA BINÁRIA
        # =========================
        mask_np = rgb_to_binary(np.array(mask))
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)

        img = self.to_tensor(img)
        img = self.normalize(img)

        return img, mask_tensor
# ==============================
# CARREGAR ARQUIVOS
# ==============================

image_files = list_all_files(base_image_tiles_dir, '.tif')
mask_files  = list_all_files(base_mask_tiles_dir , '.png')

print("Imagens:", len(image_files))
print("Máscaras:", len(mask_files))
assert len(image_files) == len(mask_files)

image_train, image_val, mask_train, mask_val = train_test_split(
    image_files, mask_files, test_size=0.3, random_state=42
)

train_dataset = TreeBinaryDataset(image_train, mask_train, img_size, train=True)
val_dataset   = TreeBinaryDataset(image_val  , mask_val  , img_size, train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

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

def loss_fn(pred, target):
    return 0.5 * bce(pred, target) + 0.5 * dice(pred, target)

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

thresholds = [0.4, 0.5, 0.6, 0.7]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=3,factor=0.5)

for epoch in range(epochs):

    # =========================
    # TREINO
    # =========================
    model.train()
    train_loss = 0

    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)

        optimizer.zero_grad()

        with autocast(enabled=(device.type == "cuda")):
            pred = model(img)
            loss = loss_fn(pred, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # =========================
    # VALIDAÇÃO + BEST THRESHOLD
    # =========================
    model.eval()
    val_loss = 0

    best_threshold = 0
    best_iou_epoch = 0

    best_metrics = None

    with torch.no_grad():
        for t in thresholds:

            TP_total = FP_total = FN_total = TN_total = 0
            val_loss_temp = 0

            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)

                with autocast(enabled=(device.type == "cuda")):
                    pred = model(img)
                    loss = loss_fn(pred, mask)

                val_loss_temp += loss.item()

                pred_bin = torch.sigmoid(pred)
                pred_bin = (pred_bin > t).float()

                pred_flat = pred_bin.view(-1)
                mask_flat = mask.view(-1)

                TP = ((pred_flat == 1) & (mask_flat == 1)).sum().item()
                FP = ((pred_flat == 1) & (mask_flat == 0)).sum().item()
                FN = ((pred_flat == 0) & (mask_flat == 1)).sum().item()
                TN = ((pred_flat == 0) & (mask_flat == 0)).sum().item()

                TP_total += TP
                FP_total += FP
                FN_total += FN
                TN_total += TN

            iou = TP_total / max(TP_total + FP_total + FN_total, 1)

            if iou > best_iou_epoch:
                best_iou_epoch = iou
                best_threshold = t

                precision = TP_total / max(TP_total + FP_total, 1)
                recall    = TP_total / max(TP_total + FN_total, 1)
                f1        = 0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall)
                acc       = (TP_total + TN_total) / max(TP_total + TN_total + FP_total + FN_total, 1)

                best_metrics = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "acc": acc,
                    "tp": TP_total,
                    "fp": FP_total,
                    "fn": FN_total,
                    "tn": TN_total,
                    "val_loss": val_loss_temp / len(val_loader),
                    "threshold": t,
                    "iou": iou
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
    print(f"Threshold escolhido: {best_threshold:.2f}")

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
        "threshold": best_threshold,
        "tp": best_metrics["tp"],
        "fp": best_metrics["fp"],
        "fn": best_metrics["fn"],
        "tn": best_metrics["tn"],
        "model": "Unet_resnet34",
        "img_size": 512,
        "batch_size": batch_size,
        "lr": 1e-4
    })

    # =========================
    # SALVAR MODELOS
    # =========================
    torch.save(model.state_dict(), checkpoint_path.replace(".pth", "_last.pth"))

    df = pd.DataFrame(history)
    df.to_csv("/home/paulomaciel/proj/lab_2026/unet_metrics_512_softAug2.csv", index=False)
    print("CSV atualizado com sucesso")
    print("CSV salvo em: /home/paulomaciel/proj/lab_2026/unet_metrics_512_softAug2.csv")

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
