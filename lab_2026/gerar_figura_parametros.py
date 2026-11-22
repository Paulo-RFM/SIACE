#!/usr/bin/env python
"""Conta os parâmetros dos modelos da monografia e gera tabela (CSV) e figura.

Somente leitura sobre runs/. Os valores vêm dos checkpoints salvos e são
conferidos contra as mesmas arquiteturas instanciadas sem pesos pré-treinados.

Saídas:
  runs/parametros_modelos_20261111/parametros.csv
  LaTex/TCC2___Paulo_Maciel (1)/parametros_modelos.png
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
from efficientnet_pytorch import EfficientNet

RAIZ = Path(__file__).resolve().parent
SAIDA_CSV = RAIZ / "runs/parametros_modelos_20261111/parametros.csv"  # 2026-11-11: + arquiteturas do benchmark
SAIDA_FIG = RAIZ / "LaTex/TCC2___Paulo_Maciel (1)/parametros_modelos.png"

BUFFERS = ("running_mean", "running_var", "num_batches_tracked")
COR_CODIFICADOR = "#2a78d6"
COR_DEMAIS = "#eb6834"
COR_TEXTO = "#0b0b0b"
COR_EIXO = "#52514e"

# rótulo, tipo, nº de saídas, runs com a mesma arquitetura
# tipo "unet" = smp.Unet ResNet-34; ("smp", arquitetura, codificador) = outras arquiteturas do benchmark
MODELOS = [
    ("SegFormer MiT-B1 (binária)", ("smp", "Segformer", "mit_b1"), 1,
     ["arq_segformer_mit_b1_20261110", "seeds_segformer_s1_20261110", "seeds_segformer_s2_20261110",
      "seeds_segformer_s3_20261110"]),
    ("U-Net ResNet-34 (binária)", "unet", 1, ["alinhado_20261105_2215_softAug4"]),
    ("U-Net++ ResNet-34 (binária)", ("smp", "UnetPlusPlus", "resnet34"), 1, ["arq_unetplusplus_resnet34_20261110"]),
    ("DeepLabV3+ ResNet-34 (binária)", ("smp", "DeepLabV3Plus", "resnet34"), 1, ["arq_deeplabv3plus_resnet34_20261110"]),
    (
        "U-Net ResNet-34 (multiclasse)",
        "unet",
        7,
        [
            "classif_20261102_163825_unet_mc_ce",
            "unet_mc_ce_dice_corrigido_20261106",
            "unet_mc_ce_dice_initbin_corrigido_20261106",
        ],
    ),
    (
        "EfficientNet-B0 (6 classes)",
        "efficientnet",
        6,
        ["classif_20261102_163825_efficientnet_anotados"],
    ),
    (
        "EfficientNet-B0 (binária)",
        "efficientnet",
        2,
        [
            "classif_20261103_115814_efficientnet_binaria_bananeira",
            "classif_20261104_235839_efficientnet_binaria_mangueira",
            "classif_20261104_235839_efficientnet_binaria_palmeira",
        ],
    ),
]


def carregar_pesos(run):
    caminho = RAIZ / "runs" / run / "checkpoint_best.pth"
    dados = torch.load(caminho, map_location="cpu", weights_only=True)
    for chave in ("model_state_dict", "state_dict", "model", "model_state", "net"):
        if isinstance(dados.get(chave), dict):
            return dados[chave]
    return dados


def contar_checkpoint(pesos, tipo):
    """Separa codificador/extrator das demais camadas, sem os buffers do BatchNorm."""
    codificador = demais = 0
    for nome, tensor in pesos.items():
        if not torch.is_tensor(tensor) or any(b in nome for b in BUFFERS):
            continue
        if tipo == "unet" or isinstance(tipo, tuple):
            eh_codificador = nome.startswith("encoder.")
        else:
            eh_codificador = not nome.startswith("_fc")
        if eh_codificador:
            codificador += tensor.numel()
        else:
            demais += tensor.numel()
    return codificador, demais


def contar_arquitetura(tipo, saidas):
    if isinstance(tipo, tuple):
        modelo = getattr(smp, tipo[1])(encoder_name=tipo[2], encoder_weights=None, in_channels=3, classes=saidas)
    elif tipo == "unet":
        modelo = smp.Unet(
            encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=saidas
        )
    else:
        modelo = EfficientNet.from_name("efficientnet-b0", num_classes=saidas)
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def milhar(valor):
    return f"{valor:,}".replace(",", ".")


def main():
    linhas = []
    for rotulo, tipo, saidas, runs in MODELOS:
        contagens = {contar_checkpoint(carregar_pesos(run), tipo) for run in runs}
        if len(contagens) != 1:
            raise SystemExit(f"Runs com contagens diferentes em {rotulo}: {contagens}")
        codificador, demais = contagens.pop()
        total = codificador + demais
        arquitetura = contar_arquitetura(tipo, saidas)
        if total != arquitetura:
            raise SystemExit(f"{rotulo}: checkpoint {total} != arquitetura {arquitetura}")
        linhas.append(
            {
                "modelo": rotulo,
                "saidas": saidas,
                "codificador": codificador,
                "demais_camadas": demais,
                "total": total,
                "runs": ";".join(runs),
            }
        )
        print(rotulo, milhar(codificador), milhar(demais), milhar(total))

    SAIDA_CSV.parent.mkdir(parents=False, exist_ok=True)
    with SAIDA_CSV.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    figura, eixo = plt.subplots(figsize=(8, 0.62 * len(linhas) + 0.6))
    posicoes = range(len(linhas))
    codificadores = [l["codificador"] / 1e6 for l in linhas]
    demais = [l["demais_camadas"] / 1e6 for l in linhas]
    eixo.barh(
        posicoes, codificadores, height=0.56, color=COR_CODIFICADOR,
        edgecolor="white", linewidth=1.2, label="Codificador",
    )
    eixo.barh(
        posicoes, demais, left=codificadores, height=0.56, color=COR_DEMAIS,
        edgecolor="white", linewidth=1.2, label="Demais camadas",
    )
    for posicao, linha in zip(posicoes, linhas):
        total = linha["total"] / 1e6
        texto = f"{total:.2f}".replace(".", ",") + " milhões"
        eixo.text(total + 0.35, posicao, texto, va="center", color=COR_TEXTO)

    eixo.set_yticks(list(posicoes), [l["modelo"] for l in linhas], color=COR_TEXTO)
    eixo.invert_yaxis()
    eixo.set_xlim(0, 35)
    eixo.set_xlabel("Número de parâmetros (milhões)", color=COR_EIXO)
    eixo.tick_params(axis="y", length=0)
    eixo.tick_params(axis="x", colors=COR_EIXO)
    eixo.grid(axis="x", color="#e4e3df", linewidth=0.8)
    eixo.set_axisbelow(True)
    for lado in ("top", "right", "left"):
        eixo.spines[lado].set_visible(False)
    eixo.spines["bottom"].set_color("#b5b4ae")
    eixo.legend(frameon=False, loc="lower right", ncol=1)
    figura.tight_layout()
    figura.savefig(SAIDA_FIG, dpi=300, facecolor="white")
    print("Figura:", SAIDA_FIG)
    print("CSV:", SAIDA_CSV)


if __name__ == "__main__":
    main()
