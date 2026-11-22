#!/usr/bin/env python3
"""Compara a SegFormer MiT-B1 treinada em 3 cm (sistema), 8 cm e 15 cm (experimento de resolução, 2026-11-21).

Modo padrão (depois dos treinos e das avaliações de teste de runs/resolucao_20261121.sh):
  - métricas de pixel no teste (IoU, F1, precisão, recall, F1 de borda) de cada modelo em cada resolução de
    teste, com IC 95% por bootstrap por tile (B = 2000, semente 42, funções de bootstrap_ic_corrigido.py);
  - diferença pareada de IoU de cada modelo contra o modelo atual, na mesma resolução de teste;
  - gráficos: curvas de treino, matriz de IoU (modelo x teste), IoU com IC e exemplo de predições.
Modo --joao-pessoa: resume as contagens dos 3 modelos na ortoimagem de João Pessoa e desenha um recorte.
Nada é treinado ou reavaliado aqui: só lê predições e saídas já salvas.
Saída: runs/resolucao_comparacao_20261121/
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from avaliar_alinhamento import MASKS, SPLIT, counts_from_pngs, metrics  # noqa: E402
from bootstrap_ic_corrigido import SEED, boot_idx, ci, iou_f1, seg_counts  # noqa: E402
from unet_spatial_protocol import load_split, paired_tiles, valid_pixel_mask  # noqa: E402

LAB = Path(__file__).resolve().parent
R = LAB / "runs"
OUT = R / "resolucao_comparacao_20261121"
RES = Path("/home/paulomaciel/proj/resolucao_20261121")
MODELOS = {"atual": ("SegFormer 3 cm (sistema)", "arq_segformer_mit_b1_20261110"),
           "8cm": ("SegFormer 8 cm", "res_segformer_8cm_20261121"),
           "15cm": ("SegFormer 15 cm", "res_segformer_15cm_20261121")}
TESTES = {"3cm": ("3 cm", Path("/home/paulomaciel/proj/imagens")),
          "8cm": ("8 cm", RES / "imagens_8cm"),
          "15cm": ("15 cm", RES / "imagens_15cm")}
PROPRIO = {"atual": "3cm", "8cm": "8cm", "15cm": "15cm"}  # resolução em que cada modelo foi treinado
CORES = {"atual": "#2f7d4a", "8cm": "#e09f3e", "15cm": "#9e2a2b"}


def eval_dir(m, t):
    if m == "atual" and t == "3cm":  # avaliação original do sistema (11/11), reaproveitada
        return R / "arquiteturas_comparacao_20261110" / "arq_segformer_mit_b1_20261110" / "test_eval"
    return OUT / "eval" / f"{m}_em_{t}"


def minutos(run_dir):
    txt = (run_dir / "log.txt").read_text(errors="ignore")
    a, b = re.search(r"started_at=(\S+)", txt), re.search(r"ended_at=(\S+?)(?:\s|$)", txt)
    return (datetime.fromisoformat(b.group(1)) - datetime.fromisoformat(a.group(1))).total_seconds() / 60 if a and b else np.nan


def comparar():
    OUT.mkdir(parents=True, exist_ok=True)
    pares = paired_tiles(str(TESTES["3cm"][1]), MASKS)
    split, _, _ = load_split(SPLIT, [i for i, _ in pares])
    img_by = {Path(i).stem: i for i, _ in pares}
    mask_by = {Path(i).stem: m for i, m in pares}
    tiles = [Path(i).stem for i in split["test"]]
    gt_by = {t: np.any(np.asarray(Image.open(mask_by[t]).convert("RGB")) != 0, axis=2) for t in tiles}
    valid_by = {t: valid_pixel_mask(Image.open(img_by[t])) for t in tiles}
    idx = boot_idx(len(tiles), np.random.default_rng(SEED))

    contagens, linhas, ic = {}, [], []
    for m, (nome, run) in MODELOS.items():
        run_dir = R / run
        limiar = json.loads((run_dir / "frozen_threshold.json").read_text())
        hist = pd.read_csv(run_dir / "metrics.csv")
        for t, (tnome, _) in TESTES.items():
            pred = eval_dir(m, t) / "predictions_test"
            c = seg_counts(pred, tiles, gt_by, valid_by)
            contagens[(m, t)] = c
            met = metrics(counts_from_pngs(pred, tiles, gt_by, valid_by))
            amostras = np.array([iou_f1(c[i]) for i in idx])
            lo, hi = ci(amostras[:, 0])
            linhas.append({"modelo": nome, "treino": PROPRIO[m], "teste": t, "mesma_resolucao": PROPRIO[m] == t,
                           "limiar": round(float(limiar["threshold"]), 2), "iou_val": limiar["validation_iou"],
                           "epocas": len(hist), "melhor_epoca": int(hist.loc[hist["iou"].idxmax(), "epoch"]),
                           "minutos_treino": round(minutos(run_dir), 1), **met,
                           "iou_ic95_inf": lo, "iou_ic95_sup": hi})
            ic.append({"comparacao": f"{nome} no teste {tnome}", "metrica": "IoU teste", "estimativa": met["iou"],
                       "ic95_inf": lo, "ic95_sup": hi, "n_tiles": len(tiles)})
    for m in ("8cm", "15cm"):
        for t in TESTES:
            a, b = contagens[(m, t)], contagens[("atual", t)]
            dif = [iou_f1(a[i])[0] - iou_f1(b[i])[0] for i in idx]
            lo, hi = ci(dif)
            ic.append({"comparacao": f"{MODELOS[m][0]} - {MODELOS['atual'][0]}, teste {TESTES[t][0]}",
                       "metrica": "dif. IoU (pareada)", "estimativa": iou_f1(a)[0] - iou_f1(b)[0],
                       "ic95_inf": lo, "ic95_sup": hi, "n_tiles": len(tiles)})
    df = pd.DataFrame(linhas)
    df.to_csv(OUT / "resultados.csv", index=False)
    pd.DataFrame(ic).to_csv(OUT / "ic_95.csv", index=False)
    mat = df.pivot(index="modelo", columns="teste", values="iou").reindex(
        [v[0] for v in MODELOS.values()])[list(TESTES)]
    mat.to_csv(OUT / "matriz_iou.csv")
    print(df[["modelo", "teste", "iou", "iou_ic95_inf", "iou_ic95_sup", "dice_f1", "precision", "recall"]]
          .round(3).to_string(index=False))
    print(pd.DataFrame(ic).round(4).to_string(index=False))

    # --- gráficos
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for m, (nome, run) in MODELOS.items():
        h = pd.read_csv(R / run / "metrics.csv")
        ax[0].plot(h.epoch, h.iou, marker="o", ms=3, color=CORES[m], label=nome)
        ax[1].plot(h.epoch, h.train_loss, color=CORES[m], label=f"{nome} (treino)")
        ax[1].plot(h.epoch, h.val_loss, color=CORES[m], ls="--", label=f"{nome} (validação)")
    ax[0].set(xlabel="Época", ylabel="IoU de validação (limiar 0,5)", title="IoU de validação por época")
    ax[1].set(xlabel="Época", ylabel="Perda", title="Perda de treino (—) e de validação (--)")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend()
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "curvas_treino.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    im = ax.imshow(mat.values, cmap="Greens", vmin=np.floor(mat.values.min() * 20) / 20, vmax=mat.values.max())
    ax.set_xticks(range(3), [f"teste {TESTES[t][0]}" for t in TESTES])
    ax.set_yticks(range(3), list(mat.index))
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat.values[i, j]:.3f}", ha="center", va="center",
                    fontweight="bold" if i == j else "normal")
    fig.colorbar(im, ax=ax, label="IoU de teste")
    ax.set_title("IoU de teste: modelo (linha) × resolução do teste (coluna)")
    fig.tight_layout()
    fig.savefig(OUT / "matriz_iou.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    for k, m in enumerate(MODELOS):
        d = df[df.treino == PROPRIO[m]].set_index("teste").loc[list(TESTES)]
        ax.errorbar(x + (k - 1) * 0.12, d.iou, yerr=[d.iou - d.iou_ic95_inf, d.iou_ic95_sup - d.iou], fmt="o",
                    capsize=4, color=CORES[m], label=MODELOS[m][0])
    ax.set_xticks(x, [f"teste {TESTES[t][0]}" for t in TESTES])
    ax.set_ylabel("IoU de teste (IC 95%)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("IoU de teste por resolução, com IC 95% (bootstrap por tile)")
    fig.tight_layout()
    fig.savefig(OUT / "iou_ic95.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for m, (nome, run) in MODELOS.items():
        cv = pd.read_csv(R / run / "threshold_iou_curve.csv")
        ax.plot(cv.threshold, cv.iou, color=CORES[m], label=nome)
    ax.set(xlabel="Limiar", ylabel="IoU de validação", title="Escolha do limiar na validação")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "curvas_limiar.png", dpi=200)
    plt.close(fig)

    t = max(tiles, key=lambda s: gt_by[s].mean() * (gt_by[s].mean() < 0.6))  # tile de teste com bastante copa
    fig, ax = plt.subplots(2, 4, figsize=(14, 8.0), gridspec_kw={"hspace": 0.35})
    ax[0, 0].imshow(gt_by[t], cmap="gray")
    ax[0, 0].set_title("Referência")
    ax[1, 0].axis("off")
    for j, (tk, (tnome, d)) in enumerate(TESTES.items()):
        ax[0, j + 1].imshow(Image.open(d / f"{t}.tif").convert("RGB"))
        ax[0, j + 1].set_title(f"Imagem {tnome}")
        m = {v: k for k, v in PROPRIO.items()}[tk]
        p = np.asarray(Image.open(eval_dir(m, tk) / "predictions_test" / f"{t}.png")) > 127
        c = contagens[(m, tk)][tiles.index(t)]
        ax[1, j + 1].imshow(p, cmap="gray")
        ax[1, j + 1].set_title(f"{MODELOS[m][0]}\nIoU do tile {c[0] / max(c.sum(), 1):.3f}")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"Tile de teste {t}: cada modelo na resolução em que foi treinado")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
    fig.savefig(OUT / "exemplo_predicoes.png", dpi=150)
    plt.close(fig)
    print("saída:", OUT)


def joao_pessoa():
    from tree_georef import read_geotransform
    base = OUT / "joao_pessoa"
    linhas = []
    for m, (nome, _) in MODELOS.items():
        s = json.loads((base / m / "summary.json").read_text())
        arv = pd.read_csv(base / m / "arvores.csv")
        linhas.append({"modelo": nome, "tiles": s["n_tiles"], "copas_nos_tiles": s["n_instances"],
                       "arvores": s["n_trees"], "area_copa_total_ha": round(arv.area_m2.sum() / 1e4, 2),
                       "area_mediana_m2": round(arv.area_m2.median(), 1), "minutos": round(s["duration_s"] / 60, 1),
                       **{f"classe_{k}": v for k, v in s["trees_by_class"].items()}})
    df = pd.DataFrame(linhas)
    df.to_csv(base / "contagens_joao_pessoa.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df.modelo, df.arvores, color=[CORES[m] for m in MODELOS])
    for i, v in enumerate(df.arvores):
        ax.text(i, v, f"{v:,}".replace(",", "."), ha="center", va="bottom")
    ax.set_ylabel("Árvores detectadas")
    ax.set_title("João Pessoa (imagem de 8 cm): árvores por modelo")
    fig.tight_layout()
    fig.savefig(base / "contagens_joao_pessoa.png", dpi=200)
    plt.close(fig)

    # recorte: o tile com mais copas no modelo de 8 cm, com as copas dos 3 modelos
    inst = pd.read_csv(base / "8cm" / "instancias.csv")
    tile = inst.tile.value_counts().index[0]
    tif = Path("/home/paulomaciel/proj_JoaoPessoa211126/tiles512_3cm_4326") / f"{tile}.tif"
    lon0, lat0, sx, sy = read_geotransform(tif)
    img = Image.open(tif).convert("RGB")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.3))
    for a, (m, (nome, _)) in zip(ax, MODELOS.items()):
        a.imshow(img)
        cop = json.loads((base / m / "copas.geojson").read_text())["features"]
        n = 0
        for f in cop:
            g = f["geometry"]
            for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
                xy = np.array(poly[0])
                x, y = (xy[:, 0] - lon0) / sx, (lat0 - xy[:, 1]) / sy
                if x.max() < 0 or x.min() > 512 or y.max() < 0 or y.min() > 512:
                    continue
                a.plot(x, y, color=CORES[m], lw=1.6)
                n += 1
        a.set_xlim(0, 512)
        a.set_ylim(512, 0)
        a.set_title(f"{nome}: {n} copas")
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"João Pessoa, tile {tile} (imagem de 8 cm na grade de 3,19 cm)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(base / "recorte_joao_pessoa.png", dpi=150)
    plt.close(fig)
    print("saída:", base)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--joao-pessoa", action="store_true")
    joao_pessoa() if ap.parse_args().joao_pessoa else comparar()
