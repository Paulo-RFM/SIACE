import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch.losses import DiceLoss

import time
from datetime import timedelta
from aligned_augmentation import paired_random_resized_crop  # correcao 2026-11-06: crop sincronizado

# ==============================
# CONFIGURAÇÕES
# ==============================

# Paths
base_image_tiles_dir = "/home/paulomaciel/proj/imagens/"
base_mask_tiles_dir  = "/home/paulomaciel/proj/lab_2026/masks_distin2/"
checkpoint_path      = "/home/paulomaciel/proj/lab_2026/unet_tree_segmentation_model3.pth"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

img_size = (512, 512)
batch_size = 6
epochs = 50
patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapeamento RGB → classe
# Paleta REAL das mascaras (make_mask_distin.py), verificada em 2026-11-02 contando as
# cores de masks_distin2/ e masks_distin/. A versao anterior usava (255,0,0) para a
# classe 1 (cor inexistente nas mascaras: a classe 1 virava fundo) e trocava 5 e 6.
color_to_class = {
    (0, 0, 0): 0,          # fundo
    (128, 128, 128): 1,    # arvore sem identificacao
    (0, 255, 0): 2,        # palmeira
    (0, 0, 255): 3,        # arvore seca
    (255, 255, 0): 4,      # mangueira
    (0, 255, 255): 5,      # bananeira
    (255, 0, 255): 6,      # jabuticabeira
}
num_classes = len(set(color_to_class.values()))

# ==============================
# UTILIDADES
# ==============================

def list_all_files(directory, extension='.png'):
    result = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(extension):
                result.append(os.path.join(root, f))
    return sorted(result)

def rgb_to_class(mask_rgb: np.ndarray, mapping: dict) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for color, cid in mapping.items():
        match = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
        class_mask[match] = cid
    return class_mask

def compute_class_weights(mask_file_list, color_to_class, img_size=None):
    counts = np.zeros(max(color_to_class.values()) + 1, dtype=np.int64)

    for mf in mask_file_list:
        m = Image.open(mf).convert("RGB")
        if img_size is not None:
            m = m.resize(img_size, Image.NEAREST)
        arr = np.array(m, dtype=np.uint8)
        cm = rgb_to_class(arr, color_to_class)
        for c in np.unique(cm):
            counts[c] += (cm == c).sum()

    counts = counts.astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv / np.mean(inv)
    return weights.tolist()

# ==============================
# DATASET
# ==============================

class AerialImageDataset(Dataset):
    def __init__(self, image_files, mask_files, img_size, train=True):
        self.image_files = image_files
        self.mask_files = mask_files
        self.img_size = img_size
        self.train = train

        self.color_jitter = transforms.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.25
        )

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.random_crop = transforms.RandomResizedCrop(
            img_size,
            scale=(0.7, 1.0),   # zoom para destacar árvores pequenas
            ratio=(1.0, 1.0)
        )

    def __len__(self):
        return len(self.image_files)

    def _apply_crop(self, image, mask):
        # Crop sincronizado de fato (correcao 2026-11-06, ver aligned_augmentation.py): mesmos parametros,
        # BILINEAR na imagem e NEAREST na mascara de classes.
        image, mask, _ = paired_random_resized_crop(image, mask, self.random_crop.size, self.random_crop.scale,
                                                    self.random_crop.ratio)
        return image, mask

    def _random_geometric(self, image, mask):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        mask  = Image.open(self.mask_files[idx]).convert("RGB")

        image = image.resize(self.img_size, Image.BILINEAR)
        mask  = mask.resize(self.img_size, Image.NEAREST)

        if self.train:
            image, mask = self._apply_crop(image, mask)
            image, mask = self._random_geometric(image, mask)
            image = self.color_jitter(image)

        image = self.to_tensor(image)
        image = self.normalize(image)

        mask_np = np.array(mask, dtype=np.uint8)
        mask_tensor = torch.tensor(rgb_to_class(mask_np, color_to_class), dtype=torch.long)

        return image, mask_tensor

# ==============================
# CARREGAMENTO DOS ARQUIVOS
# ==============================

image_files = list_all_files(base_image_tiles_dir, '.tif')
mask_files  = list_all_files(base_mask_tiles_dir , '.png')

print("Imagens:", len(image_files))
print("Máscaras:", len(mask_files))

image_train, image_val, mask_train, mask_val = train_test_split(
    image_files, mask_files, test_size=0.3, random_state=42
)

train_dataset = AerialImageDataset(image_train, mask_train, img_size, train=True)
val_dataset   = AerialImageDataset(image_val  , mask_val  , img_size, train=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ==============================
# CALCULAR PESOS DAS CLASSES (ATIVADO)
# ==============================

print("\nCalculando pesos das classes...")
# Pesos calculados SOMENTE com o treino (antes usava treino + validacao).
weights = compute_class_weights(mask_train, color_to_class, img_size=img_size)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
print("Pesos usados:", weights)

# ==============================
# MODELO + LOSS + OTIMIZADOR
# ==============================

model = Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes
).to(device)

loss_ce = nn.CrossEntropyLoss(weight=class_weights)
dice_loss = DiceLoss(mode="multiclass", from_logits=True)

def mixed_loss(pred, target):
    return 0.3 * loss_ce(pred, target) + 0.7 * dice_loss(pred, target)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=3, mode="min"
)

# ==============================
# FUNÇÃO DE MÉTRICAS
# ==============================

def segmentation_metrics_multiclass(preds, masks, num_classes):
    preds_labels = torch.argmax(preds, dim=1)
    correct = (preds_labels == masks).sum().item()
    total = masks.numel()
    accuracy = correct / total

    precisions, recalls, f1s, ious = [], [], [], []

    for c in range(1, num_classes):
        p = (preds_labels == c)
        m = (masks == c)

        TP = (p & m).sum().item()
        FP = (p & ~m).sum().item()
        FN = (~p & m).sum().item()

        precision = TP / max(TP + FP, 1)
        recall = TP / max(TP + FN, 1)
        f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        iou = TP / max(TP + FP + FN, 1)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        ious.append(iou)

    return {
        "accuracy": accuracy,
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "iou": float(np.mean(ious)),
    }


# Metricas por CONFUSAO ACUMULADA em toda a validacao (auditoria 2026-11-02).
# A versao anterior calculava precisao/recall/F1/IoU por lote e fazia a media dos
# lotes: classes ausentes em um lote entravam com 0 e distorciam as medias.
CLASS_NAMES = ["fundo", "arvore_sem_identificacao", "palmeira", "arvore_seca",
               "mangueira", "bananeira", "jabuticabeira"]


def confusion_update(cm, preds, masks, num_classes):
    pred_labels = torch.argmax(preds, dim=1).flatten()
    target = masks.flatten()
    cm += torch.bincount(num_classes * target + pred_labels,
                         minlength=num_classes ** 2).reshape(num_classes, num_classes).cpu()
    return cm


def metrics_from_confusion(cm):
    cm = cm.double()
    tp = torch.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    precision = torch.where(tp + fp > 0, tp / (tp + fp).clamp(min=1), torch.zeros_like(tp))
    recall = torch.where(tp + fn > 0, tp / (tp + fn).clamp(min=1), torch.zeros_like(tp))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall).clamp(min=1e-12),
                     torch.zeros_like(tp))
    iou = torch.where(tp + fp + fn > 0, tp / (tp + fp + fn).clamp(min=1), torch.zeros_like(tp))
    present = (cm.sum(1) > 0)
    present[0] = False  # macro sem fundo, apenas classes presentes na validacao
    out = {
        "accuracy": float(tp.sum() / cm.sum().clamp(min=1)),
        "precision": float(precision[present].mean()) if present.any() else 0.0,
        "recall": float(recall[present].mean()) if present.any() else 0.0,
        "f1": float(f1[present].mean()) if present.any() else 0.0,
        "iou": float(iou[present].mean()) if present.any() else 0.0,
    }
    for c, name in enumerate(CLASS_NAMES):
        out[f"iou_{name}"] = float(iou[c])
        out[f"f1_{name}"] = float(f1[c])
    return out

selection_metric = "iou"
loss_description = "0.3 CrossEntropy(pesos 1/freq do treino) + 0.7 Dice multiclasse"

# ==============================
# TREINAMENTO + AMP
# ==============================

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

best_iou = -1
no_improve = 0
history = []

for epoch in range(epochs):
    epoch_start = time.time()
    lr_epoch = optimizer.param_groups[0]["lr"]
    model.train()
    train_loss = 0

    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()

        with autocast():
            pred = model(img)
            loss = mixed_loss(pred, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            with autocast():
                pred = model(img)
                loss = mixed_loss(pred, mask)

            val_loss += loss.item()
            cm = confusion_update(cm, pred, mask, num_classes)

    metrics = metrics_from_confusion(cm)
    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    print(f"\n=== Época {epoch+1}/{epochs} ===")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"IoU: {metrics['iou']:.4f}")
    history.append({
        "epoch": epoch + 1,
        "lr": lr_epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        **metrics,
        "epoch_time_s": time.time() - epoch_start,
    })
    pd.DataFrame(history).to_csv(
        os.path.join(os.path.dirname(checkpoint_path), "unet_multiclass_history.csv"),
        index=False,
    )

    if metrics["iou"] > best_iou:
        best_iou = metrics["iou"]
        no_improve = 0
        torch.save(model.state_dict(), checkpoint_path)
        print("✔ Modelo atualizado (melhor IoU).")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("⛔ Early Stopping ativado.")
            break

print("\nTreinamento finalizado!")
print("Melhor IoU:", best_iou)
print("Modelo salvo em:", checkpoint_path)
