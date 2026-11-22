"""Gera figuras da monografia a partir de resultados ja salvos (somente leitura dos runs).

  matriz_confusao_especies.png <- runs/classif_20261102_163825_efficientnet_anotados/test_confusion_counts.csv
  watershed_exemplo.png        <- tile de teste: imagem, mascara predita pela SegFormer MiT-B1 (2026-11-11; antes U-Net),
                                  instancias do watershed calibrado (runs/pipeline_test_ws_segformer_20261111)
Saida: pasta do LaTeX. Os dados de cada figura tambem vao em CSV ao lado.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

LAB = Path(__file__).resolve().parent
OUT = LAB / "LaTex" / "TCC2___Paulo_Maciel (1)"
CLS_RUN = LAB / "runs" / "classif_20261102_163825_efficientnet_anotados"
E2E = LAB / "runs" / "pipeline_test_ws_segformer_20261111" / "mascaras"
LIMIAR = __import__("json").loads((LAB / "runs" / "arq_segformer_mit_b1_20261110" / "frozen_threshold.json").read_text())["threshold"]
NOMES = {"arvore_sem_identificacao": "Árvore sem\nidentificação", "palmeira": "Palmeira",
         "arvore_seca": "Árvore seca", "mangueira": "Mangueira", "bananeira": "Bananeira",
         "jabuticabeira": "Jabuticabeira"}


def matriz():
    cm = pd.read_csv(CLS_RUN / "test_confusion_counts.csv", index_col=0)
    nomes = [NOMES[c.split(":", 1)[1]] for c in cm.columns]
    m = cm.to_numpy()
    row = m / m.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(nomes)):
        for j in range(len(nomes)):
            ax.text(j, i, f"{m[i, j]}\n({row[i, j]:.2f})", ha="center", va="center", fontsize=8,
                    color="white" if row[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(nomes))); ax.set_yticks(range(len(nomes)))
    ax.set_xticklabels(nomes, rotation=45, ha="right", fontsize=9); ax.set_yticklabels(nomes, fontsize=9)
    ax.set_xlabel("Classe predita"); ax.set_ylabel("Classe anotada")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Proporção da linha (recall)")
    fig.tight_layout(); fig.savefig(OUT / "matriz_confusao_especies.png", dpi=200); plt.close(fig)
    cm.to_csv(OUT / "matriz_confusao_especies.csv")


def watershed(tile="OrtoRP-3cm_049_022"):
    img = np.asarray(Image.open(Path("/home/paulomaciel/proj/imagens") / f"{tile}.tif").convert("RGB"))
    pred = np.asarray(Image.open(E2E / "binary" / f"{tile}.png")) > 127
    inst = np.asarray(Image.open(E2E / "instances" / f"{tile}_instances.png")).astype(int)
    rng = np.random.default_rng(1)
    cores = np.vstack([[0, 0, 0], rng.uniform(0.2, 1.0, size=(inst.max(), 3))])
    over = img / 255.0 * 0.45 + cores[inst] * 0.55 * (inst > 0)[..., None] + img / 255.0 * 0.55 * (inst == 0)[..., None]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, im, t in zip(axes, [img, pred, over], ["(a) Imagem aérea", f"(b) Máscara predita (limiar {LIMIAR:.2f})".replace(".", ","),
                                                    f"(c) Copas individuais ({inst.max()})"]):
        ax.imshow(im, cmap="gray" if im is pred else None); ax.set_title(t); ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(OUT / "watershed_exemplo.png", dpi=150, bbox_inches="tight", pad_inches=0.15); plt.close(fig)
    (OUT / "watershed_exemplo.txt").write_text(f"tile={tile}\ninstancias={inst.max()}\n", encoding="utf-8")


if __name__ == "__main__":
    matriz()
    watershed()
    print("figuras salvas em", OUT)
