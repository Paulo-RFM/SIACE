import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from efficientnet_pytorch import EfficientNet

# ====================================================
# CONFIGURAÇÕES
# ====================================================

csv_path = "/home/paulomaciel/proj/species_crops_balanced/species_labels_balanced.csv"
output_dir = "/home/paulomaciel/proj/species_model/"
os.makedirs(output_dir, exist_ok=True)

num_classes = 6
batch_size = 6
epochs = 40
lr = 1e-4
img_size = 512

patience = 5        # <<< EARLY STOPPING
es_counter = 0      # contador de "épocas sem melhora"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Treinando em:", device)

# ====================================================
# DATASET
# ====================================================

class SpeciesDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["crop"]
        label = int(row["species"]) - 1

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

# ====================================================
# SPLIT
# ====================================================

df = pd.read_csv(csv_path)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["species"])

print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

# ====================================================
# TRANSFORMS
# ====================================================

train_transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(30),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

train_dataset = SpeciesDataset(train_df, train_transform)
val_dataset   = SpeciesDataset(val_df,   val_transform)

# ====================================================
# SAMPLER BALANCEADO
# ====================================================

class_counts = train_df["species"].value_counts().sort_index()
inv_freq = 1.0 / class_counts.values
weights = train_df["species"].map(lambda x: inv_freq[x-1]).values
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4)

# ====================================================
# MODELO
# ====================================================

model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# ====================================================
# TREINAMENTO
# ====================================================

best_f1 = 0
es_counter = 0

for epoch in range(1, epochs+1):

    # ========== TREINO ==========
    model.train()
    train_losses = []
    preds_train = []
    labels_train = []

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        preds_train.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_train.extend(labels.cpu().numpy())

    train_acc = accuracy_score(labels_train, preds_train)
    train_f1  = f1_score(labels_train, preds_train, average="macro")

    # ========== VALIDAÇÃO ==========
    model.eval()
    val_losses = []
    preds_val = []
    labels_val = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            val_losses.append(loss.item())
            preds_val.extend(out.argmax(1).cpu().numpy())
            labels_val.extend(labels.cpu().numpy())

    val_acc = accuracy_score(labels_val, preds_val)
    val_f1  = f1_score(labels_val, preds_val, average="macro")

    scheduler.step()

    print(f"\nEpoch {epoch}/{epochs}")
    print(f"Train Loss: {np.mean(train_losses):.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"Val   Loss: {np.mean(val_losses):.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

    # ========== CHECAR MELHORA (EARLY STOPPING) ==========
    if val_f1 > best_f1:
        best_f1 = val_f1
        es_counter = 0  # reset
        torch.save(model.state_dict(), os.path.join(output_dir, "efficientnet_b0_best.pth"))
        print("→ Novo melhor modelo! F1 =", best_f1)
    else:
        es_counter += 1
        print(f"Sem melhora por {es_counter}/{patience} épocas")

        if es_counter >= patience:
            print("\nEarly Stopping ativado!")
            break

print("\nTreinamento finalizado!")
print("Melhor F1 obtido:", best_f1)
print("Modelo salvo em:", output_dir)
