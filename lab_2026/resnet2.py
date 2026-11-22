import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from segmentation_models_pytorch import Unet

import time
from datetime import timedelta

# ==============================
# CONFIGURAÇÕES
# ==============================

# Corrigir barras e remover r-strings com '\'
base_image_tiles_dir = "/home/paulomaciel/proj/imagens/"
base_mask_tiles_dir  = "/home/paulomaciel/proj/masks_distin2/"
checkpoint_path      = "/home/paulomaciel/proj/lab_2026/unet_tree_segmentation_model_aug_bias.pth"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

img_size = (512, 512)
batch_size = 6
epochs = 50
patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
num_classes = 7  # fundo + 6 espécies

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
    # mask_rgb: H x W x 3 (uint8)
    h, w, _ = mask_rgb.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    # Para cada cor conhecida, marca os pixels dessa classe
    for color, cid in mapping.items():
        match = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
        class_mask[match] = cid
    return class_mask

# ==============================
# DATASET
# ==============================

class AerialImageDataset(Dataset):
    def __init__(self, image_files, mask_files, img_size, transform=None):
        self.image_files = image_files
        self.mask_files = mask_files
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Carrega imagem e máscara
        image = Image.open(self.image_files[idx]).convert("RGB")
        mask  = Image.open(self.mask_files[idx]).convert("RGB")  # cores distintas

        # Resize consistente: bilinear para imagem, nearest para máscara
        if self.img_size is not None:
            image = image.resize(self.img_size, Image.BILINEAR)
            mask  = mask.resize(self.img_size, Image.NEAREST)

        # ToTensor normaliza para [0,1] e rearranja para CxHxW
        image = self.transform(image) if self.transform else transforms.ToTensor()(image)

        # Converte máscara RGB para índice de classe [H, W] com valores 0..6
        mask_np = np.array(mask, dtype=np.uint8)           # H x W x 3
        class_mask = rgb_to_class(mask_np, color_to_class) # H x W
        mask_tensor = torch.tensor(class_mask, dtype=torch.long)  # long para CrossEntropy

        return image, mask_tensor

# ==============================
# CARREGAMENTO DOS ARQUIVOS
# ==============================

image_files = list_all_files(base_image_tiles_dir, '.tif')
mask_files  = list_all_files(base_mask_tiles_dir , '.png')

print("Imagens:", len(image_files))
print("Máscaras:", len(mask_files))

assert len(image_files) == len(mask_files), "Número de imagens e máscaras deve ser igual."

image_train, image_val, mask_train, mask_val = train_test_split(
    image_files, mask_files, test_size=0.3, random_state=42
)

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = AerialImageDataset(image_train, mask_train, img_size, transform)
val_dataset   = AerialImageDataset(image_val  , mask_val  , img_size, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset  , batch_size=batch_size, shuffle=False)

# ==============================
# MODELO UNET RESNET34 (MULTICLASSE)
# ==============================

model = Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=num_classes  # 7 classes
).to(device)

# CrossEntropy para multiclasse
# Opcional: pesos de classe se houver muito desbalanceamento
# class_weights = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float32).to(device)
# loss_fn = nn.CrossEntropyLoss(weight=class_weights)
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Checkpoint carregado: {checkpoint_path}")

# ==============================
# FUNÇÃO DE MÉTRICAS (MULTICLASSE)
# ==============================

def segmentation_metrics_multiclass(preds, masks, num_classes, ignore_index=0):
    # preds: [B, C, H, W]; masks: [B, H, W]
    preds_labels = torch.argmax(preds, dim=1)  # [B, H, W]

    # Acurácia global
    correct = (preds_labels == masks).sum().item()
    total   = masks.numel()
    accuracy = correct / max(total, 1)

    # Métricas por classe (1..num_classes-1), ignorando fundo (0)
    precisions, recalls, f1s, ious = [], [], [], []
    for c in range(1, num_classes):
        pred_c = (preds_labels == c)
        mask_c = (masks == c)

        TP = (pred_c & mask_c).sum().item()
        FP = (pred_c & ~mask_c).sum().item()
        FN = (~pred_c & mask_c).sum().item()

        precision = TP / max((TP + FP), 1)
        recall    = TP / max((TP + FN), 1)
        f1        = 0.0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall)
        iou       = TP / max((TP + FP + FN), 1)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        ious.append(iou)

    # Macro médias (somente classes > 0)
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall    = float(np.mean(recalls   )) if recalls    else 0.0
    macro_f1        = float(np.mean(f1s       )) if f1s        else 0.0
    mean_iou        = float(np.mean(ious      )) if ious       else 0.0

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "iou": mean_iou
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

selection_metric = "val_loss"
loss_description = "CrossEntropyLoss (sem pesos)"

# ==============================
# TREINAMENTO + ETA + MÉTRICAS
# ==============================

best_loss = float('inf')
no_improve = 0
epoch_times = []
history = []

for epoch in range(epochs):
    start_time = time.time()

    # ===== TREINO =====
    model.train()
    train_loss = 0.0
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)  # img: [B,3,H,W]; mask: [B,H,W]

        pred = model(img)  # [B, C, H, W]
        loss = loss_fn(pred, mask)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= max(len(train_loader), 1)

    # ===== VALIDAÇÃO =====
    model.eval()
    val_loss = 0.0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = loss_fn(pred, mask)
            val_loss += loss.item()

            cm = confusion_update(cm, pred, mask, num_classes)

    val_loss /= max(len(val_loader), 1)
    metric_sum = metrics_from_confusion(cm)

    # ===== TEMPO / ETA =====
    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)
    avg_time = sum(epoch_times) / len(epoch_times)
    eta = timedelta(seconds=int(avg_time * (epochs - epoch - 1)))

    # ===== LOG =====
    print("\n===================================")
    print(f" ÉPOCA {epoch+1}/{epochs} | Tempo: {epoch_time:.2f}s | ETA: {eta}")
    print("-----------------------------------")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print("-----------------------------------")
    print(f"Accuracy : {metric_sum['accuracy']:.4f}")
    print(f"Precision: {metric_sum['precision']:.4f}  (macro, sem fundo)")
    print(f"Recall   : {metric_sum['recall']:.4f}     (macro, sem fundo)")
    print(f"F1-score : {metric_sum['f1']:.4f}        (macro, sem fundo)")
    print(f"Mean IoU : {metric_sum['iou']:.4f}       (sem fundo)")
    print("===================================\n")

    history.append({
        "epoch": epoch + 1,
        "lr": optimizer.param_groups[0]["lr"],
        "train_loss": train_loss,
        "val_loss": val_loss,
        **metric_sum,
        "epoch_time_s": epoch_time,
    })
    pd.DataFrame(history).to_csv(
        os.path.join(os.path.dirname(checkpoint_path), "unet_multiclass_bias_history.csv"),
        index=False,
    )

    # ===== EARLY STOPPING =====
    if val_loss < best_loss:
        best_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), checkpoint_path)
        print("Melhor modelo salvo!")
    else:
        no_improve += 1
        print(f"Sem melhora: {no_improve}/{patience}")
        if no_improve >= patience:
            print("Early stopping ativado.")
            break

print("\nTreinamento finalizado!")
print(f"✔ Melhor modelo salvo em: {checkpoint_path}")
