"""
gerar_matrizes_confusao.py

Gera matrizes de confusão para os dois modelos do pipeline:

  1. EfficientNet-B0 — classificação de espécies (6×6)
  2. Modelos U-Net  — segmentação binária (2×2, pixel-level)
       • modelos principais (256/512 px, semAug/softAug)
       • modelos do benchmark de encoders (models_benchmark/)

Saída: LaTex/  (prontos para \includegraphics no main.tex)
"""

import os
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from segmentation_models_pytorch import Unet
from efficientnet_pytorch import EfficientNet

# ============================================================
# CONFIGURAÇÕES — ajuste aqui se necessário
# ============================================================

BASE  = Path("/home/paulomaciel/proj")
LATEX = BASE / "LaTex"
LATEX.mkdir(parents=True, exist_ok=True)

DPI = 300

# --- Dados ---
IMAGES_DIR  = BASE / "imagens"
MASKS_DIR   = BASE / "masks"
SPECIES_CSV = BASE / "species_crops_balanced" / "species_labels_balanced.csv"

# --- Modelos principais da U-Net ---
# (encoder, caminho do checkpoint, label curto para o gráfico)
UNET_MODELS = [
    ("resnet34", BASE / "unet_binary_tree_detector256_semAug.pth",  "256px\nsem aug"),
    ("resnet34", BASE / "unet_binary_tree_detector256_softAug.pth", "256px\naug suave"),
    ("resnet34", BASE / "unet_binary_tree_detector512_softAug.pth", "512px\naug suave\n(run 1)"),
    ("resnet34", BASE / "unet_binary_tree_detector512_softAug2.pth","512px\naug suave\n(run 2)"),
]

# --- Benchmark de encoders ---
BENCHMARK_DIR     = BASE / "models_benchmark"
BENCHMARK_CSV     = BASE / "benchmark_unet.csv"

# --- Classificador de espécies ---
SPECIES_MODEL_PATH = BASE / "species_model" / "efficientnet_b0_best.pth"
NUM_CLASSES        = 6

# Nomes das espécies — substitua pelos nomes reais se souber
SPECIES_NAMES = [
    "Arovore sem identificação",  # classe 0 (sem espécie identificada)
    "Palmeira",
    "Arvore seca",
    "Mangueira",
    "Bananeira",
    "Jabuticabeira",
]

# Thresholds ótimos por configuração (do benchmark_unet.csv)
# Usado na avaliação dos modelos principais
THRESHOLD_DEFAULT = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ============================================================
# UTILITÁRIOS
# ============================================================

def list_files(directory, ext):
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if f.lower().endswith(ext):
                files.append(os.path.join(root, f))
    return sorted(files)


def build_pairs(image_files, mask_files):
    """Emparelha imagens e máscaras pelo stem do nome."""
    img_dict  = {Path(f).stem: f for f in image_files}
    mask_dict = {Path(f).stem: f for f in mask_files}
    common = sorted(set(img_dict) & set(mask_dict))
    return [img_dict[k] for k in common], [mask_dict[k] for k in common]


def rgb_to_binary(mask_rgb):
    return np.any(mask_rgb != [0, 0, 0], axis=-1).astype(np.uint8)


# ============================================================
# DATASETS
# ============================================================

class SegDataset(Dataset):
    """Dataset de segmentação binária (mesma lógica dos scripts de treino)."""
    def __init__(self, img_files, mask_files, img_size=(512, 512)):
        self.img_files  = img_files
        self.mask_files = mask_files
        self.img_size   = img_size
        self.to_tensor  = transforms.ToTensor()
        self.normalize  = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img  = Image.open(self.img_files[idx]).convert("RGB").resize(self.img_size, Image.BILINEAR)
        mask = Image.open(self.mask_files[idx]).convert("RGB").resize(self.img_size, Image.NEAREST)

        mask_bin = rgb_to_binary(np.array(mask))
        mask_t   = torch.tensor(mask_bin, dtype=torch.float32).unsqueeze(0)
        img_t    = self.normalize(self.to_tensor(img))
        return img_t, mask_t


class SpeciesDataset(Dataset):
    """Dataset de classificação de espécies (mesma lógica do treino)."""
    def __init__(self, df, img_size=512):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row["crop"]).convert("RGB")
        label = int(row["species"]) - 1   # 1-6 → 0-5
        return self.transform(img), label


# ============================================================
# FUNÇÕES DE AVALIAÇÃO
# ============================================================

def eval_unet_binary(model, val_loader, threshold=0.5):
    """
    Retorna arrays 1D de y_true e y_pred (nível de pixel)
    para construção da matriz de confusão 2×2.
    """
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs  = imgs.to(DEVICE)
            preds = torch.sigmoid(model(imgs))
            preds = (preds > threshold).float().cpu()

            all_true.append(masks.flatten().numpy().astype(np.int32))
            all_pred.append(preds.flatten().numpy().astype(np.int32))

    return np.concatenate(all_true), np.concatenate(all_pred)


def eval_species(model, val_loader):
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            out  = model(imgs)
            pred = out.argmax(dim=1).cpu().numpy()
            all_true.extend(labels.numpy())
            all_pred.extend(pred)

    return np.array(all_true), np.array(all_pred)


# ============================================================
# FUNÇÕES DE PLOT
# ============================================================

CMAP_BLUE = LinearSegmentedColormap.from_list(
    "blue_white", ["#ffffff", "#2166ac"]
)

def plot_confusion_matrix(cm, class_names, title, out_path,
                           normalize=True, figsize=(8, 6)):
    """Plota e salva uma matriz de confusão com seaborn."""
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".1%"
        vmax = 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmax = cm.max()

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap=CMAP_BLUE,
        vmin=0, vmax=vmax,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="lightgray",
        ax=ax,
        annot_kws={"size": 10},
    )

    ax.set_xlabel("Predito",  fontsize=12, labelpad=8)
    ax.set_ylabel("Real",     fontsize=12, labelpad=8)
    ax.set_title(title,       fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x",  labelsize=10, rotation=30)
    ax.tick_params(axis="y",  labelsize=10, rotation=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {Path(out_path).name}")


def plot_unet_comparison(results, out_path):
    """
    Plota barras comparando IoU / F1 / Precisão / Recall
    entre múltiplos modelos U-Net.
    """
    labels   = [r["label"]    for r in results]
    ious     = [r["iou"]      for r in results]
    f1s      = [r["f1"]       for r in results]
    precs    = [r["precision"] for r in results]
    recalls  = [r["recall"]   for r in results]

    x    = np.arange(len(labels))
    w    = 0.2
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 5))

    ax.bar(x - 1.5*w, ious,    width=w, label="IoU",      color="#2166ac")
    ax.bar(x - 0.5*w, f1s,     width=w, label="F1-Score", color="#1a9641")
    ax.bar(x + 0.5*w, precs,   width=w, label="Precisão", color="#d6604d")
    ax.bar(x + 1.5*w, recalls, width=w, label="Recall",   color="#756bb1")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.60, 1.00)
    ax.set_ylabel("Valor da Métrica")
    ax.set_title(
        "Comparativo de Métricas — Modelos U-Net (validação)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9, ncol=4, loc="lower right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2, rotation=90)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {Path(out_path).name}")


# ============================================================
# BLOCO 1 — MODELOS U-NET PRINCIPAIS
# ============================================================

def avaliar_unet_principais():
    print("\n" + "="*60)
    print("BLOCO 1 — Modelos U-Net principais")
    print("="*60)

    # Dados de validação (mesmo split do treino: 70/30, seed=42)
    images = list_files(IMAGES_DIR, ".tif")
    masks  = list_files(MASKS_DIR,  ".png")
    images, masks = build_pairs(images, masks)

    _, img_val, _, mask_val = train_test_split(
        images, masks, test_size=0.3, random_state=42
    )

    results = []

    for encoder, ckpt, label in UNET_MODELS:
        if not Path(ckpt).exists():
            print(f"[SKIP] Checkpoint não encontrado: {ckpt.name}")
            continue

        print(f"\n→ Avaliando: {ckpt.name}")

        # Detectar img_size pelo nome do arquivo
        img_size = (256, 256) if "256" in ckpt.name else (512, 512)

        val_ds     = SegDataset(img_val, mask_val, img_size=img_size)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

        model = Unet(
            encoder_name=encoder,
            encoder_weights=None,   # não baixa pesos — já carregamos do .pth
            in_channels=3,
            classes=1,
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

        y_true, y_pred = eval_unet_binary(model, val_loader, threshold=THRESHOLD_DEFAULT)

        cm  = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        iou       = tp / max(tp + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)

        print(f"   IoU={iou:.4f}  F1={f1:.4f}  Prec={precision:.4f}  Rec={recall:.4f}")

        # Matriz de confusão individual
        plot_confusion_matrix(
            cm,
            class_names=["Fundo", "Árvore"],
            title=f"Matriz de Confusão U-Net\n{label.replace(chr(10), ', ')}",
            out_path=LATEX / f"cm_unet_{ckpt.stem}.png",
            normalize=True,
            figsize=(4.5, 4),
        )

        results.append({
            "label":     label,
            "iou":       iou,
            "f1":        f1,
            "precision": precision,
            "recall":    recall,
        })

    if results:
        plot_unet_comparison(results, LATEX / "cm_unet_comparativo.png")


# ============================================================
# BLOCO 2 — BENCHMARK DE ENCODERS (models_benchmark/)
# ============================================================

def avaliar_benchmark_encoders():
    print("\n" + "="*60)
    print("BLOCO 2 — Benchmark de encoders (models_benchmark/)")
    print("="*60)

    bench_models = sorted(glob.glob(str(BENCHMARK_DIR / "*.pth")))
    if not bench_models:
        print("[SKIP] Nenhum modelo encontrado em models_benchmark/")
        return

    # Lê os thresholds ótimos do CSV
    bench_df = pd.read_csv(BENCHMARK_CSV) if BENCHMARK_CSV.exists() else None

    images = list_files(IMAGES_DIR, ".tif")
    masks  = list_files(MASKS_DIR,  ".png")
    images, masks = build_pairs(images, masks)

    _, img_val, _, mask_val = train_test_split(
        images, masks, test_size=0.3, random_state=42
    )

    val_ds     = SegDataset(img_val, mask_val, img_size=(512, 512))
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    results = []

    for ckpt_path in bench_models:
        name = Path(ckpt_path).stem   # ex: "resnet34_lr1e-04_iou0.706"

        # Inferir encoder do nome do arquivo
        encoder = name.split("_lr")[0]   # tudo antes de "_lr..."

        # Inferir threshold ótimo do CSV se possível
        threshold = THRESHOLD_DEFAULT
        if bench_df is not None:
            match = bench_df[bench_df["encoder"] == encoder]
            if not match.empty:
                best_row = match.loc[match["best_iou"].idxmax()]
                threshold = float(best_row["best_threshold"])

        print(f"\n→ {name}  encoder={encoder}  threshold={threshold}")

        try:
            model = Unet(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=1,
            ).to(DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        except Exception as e:
            print(f"   [ERRO ao carregar modelo] {e}")
            continue

        y_true, y_pred = eval_unet_binary(model, val_loader, threshold=threshold)

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        iou       = tp / max(tp + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)

        print(f"   IoU={iou:.4f}  F1={f1:.4f}  Prec={precision:.4f}  Rec={recall:.4f}")

        safe_name = name.replace(".", "_")
        plot_confusion_matrix(
            cm,
            class_names=["Fundo", "Árvore"],
            title=f"Matriz de Confusão — {encoder}",
            out_path=LATEX / f"cm_bench_{safe_name}.png",
            normalize=True,
            figsize=(4.5, 4),
        )

        results.append({
            "label":     encoder.replace("_", "\n"),
            "iou":       iou,
            "f1":        f1,
            "precision": precision,
            "recall":    recall,
        })

    if results:
        plot_unet_comparison(results, LATEX / "cm_bench_comparativo.png")


# ============================================================
# BLOCO 3 — CLASSIFICADOR DE ESPÉCIES (EfficientNet-B0)
# ============================================================

def avaliar_especies():
    print("\n" + "="*60)
    print("BLOCO 3 — EfficientNet-B0 (classificação de espécies)")
    print("="*60)

    if not SPECIES_MODEL_PATH.exists():
        print(f"[SKIP] Modelo não encontrado: {SPECIES_MODEL_PATH}")
        return

    if not SPECIES_CSV.exists():
        print(f"[SKIP] CSV não encontrado: {SPECIES_CSV}")
        return

    df = pd.read_csv(SPECIES_CSV)
    print(f"Total de amostras no CSV: {len(df)}")
    print(df["species"].value_counts().sort_index())

    # Mesmo split do treino: 80/20, estratificado, seed=42
    _, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["species"]
    )
    print(f"Validação: {len(val_df)} amostras")

    val_ds     = SpeciesDataset(val_df)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

    model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(SPECIES_MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)

    y_true, y_pred = eval_species(model, val_loader)

    cm = confusion_matrix(y_true, y_pred)

    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=SPECIES_NAMES, digits=4))

    # Acurácia e F1 macro
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"Acurácia: {acc:.4f}   F1-macro: {f1:.4f}")

    # Matriz 6×6 normalizada
    plot_confusion_matrix(
        cm,
        class_names=SPECIES_NAMES,
        title="Matriz de Confusão — EfficientNet-B0\nClassificação de Espécies Arbóreas",
        out_path=LATEX / "matriz_confusao_especies.png",
        normalize=True,
        figsize=(8, 7),
    )

    # Versão com contagens absolutas (útil para análise)
    plot_confusion_matrix(
        cm,
        class_names=SPECIES_NAMES,
        title="Matriz de Confusão (contagens) — EfficientNet-B0\nClassificação de Espécies Arbóreas",
        out_path=LATEX / "matriz_confusao_especies_abs.png",
        normalize=False,
        figsize=(8, 7),
    )

    # Imprimir métricas por classe para preencher a Tabela LaTeX
    print("\n--- Métricas por espécie (para tabela LaTeX) ---")
    from sklearn.metrics import precision_score, recall_score
    for i, name in enumerate(SPECIES_NAMES):
        mask_i = (y_true == i)
        prec_i = precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        rec_i  = recall_score   (y_true, y_pred, labels=[i], average="macro", zero_division=0)
        f1_i   = f1_score       (y_true, y_pred, labels=[i], average="macro", zero_division=0)
        n      = mask_i.sum()
        print(f"  {name}: N={n}  Prec={prec_i:.4f}  Rec={rec_i:.4f}  F1={f1_i:.4f}")


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    plt.rcParams.update({
        "font.family":    "serif",
        "font.size":      11,
        "axes.titlesize": 12,
    })

    avaliar_unet_principais()
    avaliar_benchmark_encoders()
    avaliar_especies()

    print(f"\n✔ Todos os arquivos salvos em: {LATEX.resolve()}")
