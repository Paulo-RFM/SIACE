"""Gera curvas_treinamento_seg.png (monografia, fig:curvas_seg) a partir do modelo de segmentacao do sistema.

2026-11-11: SegFormer MiT-B1 (substituiu a U-Net ResNet-34; antes: runs/alinhado_20261105_2215_softAug4).
Fonte unica: runs/arq_segformer_mit_b1_20261110/metrics.csv (somente leitura). Ate 2026-11-06 usava o
run p1_20261101_140519, treinado com o crop desalinhado (ver CHANGELOG.md).
Saidas no diretorio do LaTeX: curvas_treinamento_seg.png e curvas_treinamento_seg.csv
(dados exatamente plotados). As metricas por epoca sao de VALIDACAO, limiar 0,5
(o limiar final foi selecionado depois, na validacao, e congelado para o teste).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LAB = Path(__file__).resolve().parent
RUN = LAB / "runs" / "arq_segformer_mit_b1_20261110"
OUT = LAB / "LaTex" / "TCC2___Paulo_Maciel (1)"

m = pd.read_csv(RUN / "metrics.csv")
cols = ["epoch", "train_loss", "val_loss", "f1", "iou", "threshold"]
m[cols].to_csv(OUT / "curvas_treinamento_seg.csv", index=False)
best = m.loc[m["iou"].idxmax()]

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
a1.plot(m["epoch"], m["train_loss"], "o-", label="treino")
a1.plot(m["epoch"], m["val_loss"], "s-", label="validação")
a1.set_ylabel("Perda (0,5 BCE + 0,5 Dice)"); a1.legend(); a1.grid(alpha=0.3)
a2.plot(m["epoch"], m["f1"], "o-", label="F1 (validação)")
a2.plot(m["epoch"], m["iou"], "s-", label="IoU (validação)")
a2.axvline(best["epoch"], color="gray", ls="--", lw=0.8,
           label=f"checkpoint selecionado (época {int(best['epoch'])})")
a2.set_xlabel("Época"); a2.set_ylabel("Métrica (limiar 0,5)"); a2.legend(); a2.grid(alpha=0.3)
fig.suptitle("SegFormer MiT-B1")
fig.tight_layout()
fig.savefig(OUT / "curvas_treinamento_seg.png", dpi=200)
print("melhor epoca:", int(best["epoch"]), "IoU val:", round(best["iou"], 4))
