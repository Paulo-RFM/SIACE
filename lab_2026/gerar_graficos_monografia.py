"""
Gera os gráficos para o capítulo de Resultados da monografia.
Saída: pasta LaTex/ (onde o LaTeX procura as imagens)

Figuras geradas:
  - curvas_treinamento_seg.png   -> \ref{fig:curvas_seg}
  - benchmark_encoders.png       -> figura opcional para reforçar a Tabela 1
  - comparativo_configuracoes.png -> figura opcional para reforçar a Tabela 2
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# -------------------------------------------------------
# CONFIGURAÇÕES
# -------------------------------------------------------

BASE   = Path(__file__).parent          # pasta tcc2/
LATEX  = BASE / "LaTex"                # onde salvar as figuras

DPI    = 300
STYLE  = "seaborn-v0_8-whitegrid"      # fundo limpo para impressão acadêmica

# Paleta de cores acessível (daltonismo-friendly)
COR_TREINO = "#2166ac"   # azul escuro
COR_VAL    = "#d6604d"   # vermelho-laranja
COR_F1     = "#1a9641"   # verde
COR_IOU    = "#756bb1"   # roxo

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       100,
})

# -------------------------------------------------------
# FIGURA 1 — Curvas de treinamento do melhor modelo de segmentação
# Fonte: unet_metrics_512_softAug.csv
# -------------------------------------------------------

def plot_curvas_treinamento():
    csv = BASE / "unet_metrics_512_softAug.csv"
    df  = pd.read_csv(csv)

    # Melhor época por IoU
    best_idx = df["iou"].idxmax()
    best_ep  = df.loc[best_idx, "epoch"]
    best_iou = df.loc[best_idx, "iou"]
    best_f1  = df.loc[best_idx, "f1"]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle(
        "Curvas de Treinamento — U-Net ResNet-34 (512 px, aug. suave)",
        fontsize=12, fontweight="bold", y=1.01
    )

    epochs = df["epoch"]

    # --- Subplot 1: Perda ---
    ax = axes[0]
    ax.plot(epochs, df["train_loss"], color=COR_TREINO, lw=2, label="Perda de treino")
    ax.plot(epochs, df["val_loss"],   color=COR_VAL,    lw=2, label="Perda de validação",
            linestyle="--")
    ax.axvline(best_ep, color="gray", lw=1, linestyle=":", alpha=0.8)
    ax.set_ylabel("Perda (Loss)")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # --- Subplot 2: F1 e IoU ---
    ax = axes[1]
    ax.plot(epochs, df["f1"],  color=COR_F1,  lw=2, label="F1-Score")
    ax.plot(epochs, df["iou"], color=COR_IOU, lw=2, label="IoU", linestyle="--")
    ax.axvline(best_ep, color="gray", lw=1, linestyle=":",  alpha=0.8,
               label=f"Melhor época ({int(best_ep)})")
    ax.annotate(
        f"IoU = {best_iou:.3f}\nF1 = {best_f1:.3f}",
        xy=(best_ep, best_iou),
        xytext=(best_ep + 1.5, best_iou - 0.04),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
        color="gray"
    )
    ax.set_xlabel("Época")
    ax.set_ylabel("Métrica")
    ax.set_ylim(0.65, 0.95)
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    out = LATEX / "curvas_treinamento_seg.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out.name}")


# -------------------------------------------------------
# FIGURA 2 — Comparativo de encoders (benchmark)
# Fonte: benchmark_unet.csv
# -------------------------------------------------------

def plot_benchmark_encoders():
    csv = BASE / "benchmark_unet.csv"
    df  = pd.read_csv(csv)

    # Rótulos mais legíveis
    df["label"] = df["encoder"].str.replace("_", "\n") + "\nlr=" + df["lr"].astype(str)

    # Cores: melhor configuração em destaque
    cores = [COR_TREINO if i != df["best_iou"].idxmax() else "#d73027"
             for i in df.index]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars = ax.bar(range(len(df)), df["best_iou"], color=cores, edgecolor="white", width=0.6)

    # Rótulo de valor em cima de cada barra
    for bar, val in zip(bars, df["best_iou"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], fontsize=9)
    ax.set_ylabel("Melhor IoU (validação)")
    ax.set_ylim(0.68, 0.74)
    ax.set_title(
        "Comparativo de Encoders — U-Net (512 px)",
        fontsize=12, fontweight="bold"
    )

    # Legenda manual
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d73027", label="Melhor configuração"),
        Patch(facecolor=COR_TREINO, label="Demais configurações"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    out = LATEX / "benchmark_encoders.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out.name}")


# -------------------------------------------------------
# FIGURA 3 — Comparativo de configurações (resolução + aug)
# Fontes: unet_metrics.csv, unet_metrics_256_softAug.csv,
#         unet_metrics_512_softAug.csv, unet_metrics_512_softAug2.csv
# -------------------------------------------------------

def plot_comparativo_configuracoes():
    configs = [
        {
            "csv":    BASE / "unet_metrics.csv",
            "label":  "256 px\nsem aug",
            "iou_col": "iou",
        },
        {
            "csv":    BASE / "unet_metrics_256_softAug.csv",
            "label":  "256 px\naug suave",
            "iou_col": "iou",
        },
        {
            "csv":    BASE / "unet_metrics_512_softAug.csv",
            "label":  "512 px\naug suave\n(run 1)",
            "iou_col": "iou",
        },
        {
            "csv":    BASE / "unet_metrics_512_softAug2.csv",
            "label":  "512 px\naug suave\n(run 2)",
            "iou_col": "iou",
        },
    ]

    labels   = []
    best_ious = []

    for c in configs:
        df = pd.read_csv(c["csv"])
        best_ious.append(df[c["iou_col"]].max())
        labels.append(c["label"])

    # Cores: melhor em destaque
    best_idx = int(np.argmax(best_ious))
    cores = [COR_TREINO] * len(labels)
    cores[best_idx] = "#d73027"

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars = ax.bar(range(len(labels)), best_ious, color=cores, edgecolor="white", width=0.55)

    for bar, val in zip(bars, best_ious):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Melhor IoU (validação)")
    ax.set_ylim(0.69, 0.77)
    ax.set_title(
        "Impacto da Resolução e do Aumento de Dados\nU-Net ResNet-34",
        fontsize=12, fontweight="bold"
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d73027", label="Melhor configuração"),
        Patch(facecolor=COR_TREINO, label="Demais configurações"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    out = LATEX / "comparativo_configuracoes.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out.name}")


# -------------------------------------------------------
# FIGURA 4 — Curvas de todas as configurações sobrepostas (IoU)
# Útil para comparar convergência entre configurações
# -------------------------------------------------------

def plot_curvas_iou_comparadas():
    arquivos = {
        "256 px, sem aug":        BASE / "unet_metrics.csv",
        "256 px, aug. suave":     BASE / "unet_metrics_256_softAug.csv",
        "512 px, aug. suave (run 1)": BASE / "unet_metrics_512_softAug.csv",
        "512 px, aug. suave (run 2)": BASE / "unet_metrics_512_softAug2.csv",
    }

    cores_linha = ["#1a9641", "#2166ac", "#d6604d", "#756bb1"]
    estilos     = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for (label, csv), cor, ls in zip(arquivos.items(), cores_linha, estilos):
        df = pd.read_csv(csv)
        col_iou = "iou"
        ax.plot(range(1, len(df) + 1), df[col_iou],
                color=cor, linestyle=ls, lw=2, label=label)

    ax.set_xlabel("Época")
    ax.set_ylabel("IoU (validação)")
    ax.set_title(
        "Comparativo de IoU por Época — Configurações de Segmentação",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    out = LATEX / "curvas_iou_comparadas.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out.name}")


# -------------------------------------------------------
# EXECUÇÃO
# -------------------------------------------------------

if __name__ == "__main__":
    try:
        plt.style.use(STYLE)
    except OSError:
        plt.style.use("seaborn-whitegrid")

    LATEX.mkdir(parents=True, exist_ok=True)
    print("Gerando gráficos para a monografia...\n")

    plot_curvas_treinamento()
    plot_benchmark_encoders()
    plot_comparativo_configuracoes()
    plot_curvas_iou_comparadas()

    print(f"\nTodos os arquivos salvos em: {LATEX.resolve()}")
