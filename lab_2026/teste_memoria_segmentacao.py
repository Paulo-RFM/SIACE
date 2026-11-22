#!/usr/bin/env python
"""Mede memória da GPU e determinismo para as arquiteturas de segmentação candidatas.

Reproduz as condições do treino da U-Net ResNet-34 (tree_segmentation_resnet_26_512softAug4.py):
imagens 512x512, lote 6, AdamW, precisão mista (autocast + GradScaler), perda 0,5 BCE + 0,5 Dice e
torch.use_deterministic_algorithms(True). Usa dados aleatórios e pesos não pré-treinados: nada é
treinado de fato e nenhum checkpoint é lido ou gravado.

Saída: runs/teste_memoria_segmentacao_20261110/memoria.csv
"""
import csv
import gc
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

SAIDA = Path(__file__).resolve().parent / "runs/teste_memoria_segmentacao_20261110/memoria.csv"
LADO = 512
PASSOS = 3
CASOS = [
    ("Unet", "resnet34"),
    ("UnetPlusPlus", "resnet34"),
    ("DeepLabV3Plus", "resnet34"),
    ("Segformer", "mit_b2"),
    ("Segformer", "mit_b1"),
]


def construir(arquitetura, codificador):
    classe = getattr(smp, arquitetura)
    return classe(encoder_name=codificador, encoder_weights=None, in_channels=3, classes=1)


def medir(arquitetura, codificador, lote, deterministico, dispositivo):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(deterministico)
    modelo = None
    try:
        modelo = construir(arquitetura, codificador).to(dispositivo).train()
        parametros = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
        otimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-4, weight_decay=1e-4)
        escalador = torch.amp.GradScaler("cuda")
        dice = DiceLoss(mode="binary", from_logits=True)
        for _ in range(PASSOS):
            imagens = torch.randn(lote, 3, LADO, LADO, device=dispositivo)
            alvo = (torch.rand(lote, 1, LADO, LADO, device=dispositivo) > 0.7).float()
            otimizador.zero_grad()
            with torch.autocast("cuda"):
                saida = modelo(imagens)
                perda = 0.5 * nn.functional.binary_cross_entropy_with_logits(saida, alvo) + 0.5 * dice(saida, alvo)
            escalador.scale(perda).backward()
            escalador.step(otimizador)
            escalador.update()
        if tuple(saida.shape[-2:]) != (LADO, LADO):
            return parametros, f"saida_{tuple(saida.shape)}", "", ""
        return (
            parametros,
            "ok",
            round(torch.cuda.max_memory_allocated() / 2**20),
            round(torch.cuda.max_memory_reserved() / 2**20),
        )
    except RuntimeError as erro:
        texto = str(erro)
        if "out of memory" in texto.lower():
            return "", "sem_memoria", "", ""
        if "deterministic" in texto.lower():
            return "", "nao_deterministico: " + texto.split("\n")[0][:160], "", ""
        raise
    finally:
        del modelo
        gc.collect()
        torch.cuda.empty_cache()
        torch.use_deterministic_algorithms(False)


def main():
    dispositivo = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory // 2**20
    print("GPU:", torch.cuda.get_device_name(0), total, "MiB", flush=True)
    linhas = []
    for arquitetura, codificador in CASOS:
        for lote, deterministico in ((6, True), (6, False)):
            parametros, estado, alocada, reservada = medir(arquitetura, codificador, lote, deterministico, dispositivo)
            linhas.append({
                "arquitetura": arquitetura, "codificador": codificador, "lado_imagem": LADO, "lote": lote,
                "deterministico": deterministico, "parametros": parametros, "estado": estado,
                "pico_alocado_mib": alocada, "pico_reservado_mib": reservada, "gpu_total_mib": total,
            })
            print(linhas[-1], flush=True)
    SAIDA.parent.mkdir(parents=False, exist_ok=False)
    with SAIDA.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)
    print("CSV:", SAIDA)


if __name__ == "__main__":
    main()
