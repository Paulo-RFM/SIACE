#!/usr/bin/env python
"""Mede o pico de memória da GPU em passos de treino da EfficientNet.

Usa dados aleatórios e pesos não pré-treinados: mede só a memória, sem treinar
nenhum modelo e sem ler ou alterar checkpoints. Sem precisão mista, como em
train_species_efficientnet.py. Para cada caso, aumenta o lote até faltar memória.

Saída: runs/teste_memoria_efficientnet_20261110/memoria.csv
"""
import csv
import gc
from pathlib import Path

import torch
from efficientnet_pytorch import EfficientNet

SAIDA = Path(__file__).resolve().parent / "runs/teste_memoria_efficientnet_20261110/memoria.csv"
CASOS = [("b0", 512, 6, 6), ("b3", 300, 1, 12), ("b3", 512, 1, 6), ("b5", 456, 1, 4), ("b5", 512, 1, 4)]
PASSOS = 3
CLASSES = 6


def medir(variante, lado, lote, dispositivo):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    modelo = None
    try:
        modelo = EfficientNet.from_name(f"efficientnet-{variante}", num_classes=CLASSES)
        modelo = modelo.to(dispositivo).train()
        otimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-4)
        for _ in range(PASSOS):
            imagens = torch.randn(lote, 3, lado, lado, device=dispositivo)
            rotulos = torch.randint(0, CLASSES, (lote,), device=dispositivo)
            otimizador.zero_grad()
            torch.nn.functional.cross_entropy(modelo(imagens), rotulos).backward()
            otimizador.step()
        return (
            "ok",
            round(torch.cuda.max_memory_allocated() / 2**20),
            round(torch.cuda.max_memory_reserved() / 2**20),
        )
    except RuntimeError as erro:
        if "out of memory" in str(erro).lower():
            return "sem_memoria", "", ""
        raise
    finally:
        del modelo
        gc.collect()
        torch.cuda.empty_cache()


def main():
    dispositivo = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory // 2**20
    print("GPU:", torch.cuda.get_device_name(0), total, "MiB")
    linhas = []
    for variante, lado, lote_inicial, lote_final in CASOS:
        for lote in range(lote_inicial, lote_final + 1):
            estado, alocada, reservada = medir(variante, lado, lote, dispositivo)
            linhas.append(
                {
                    "modelo": f"efficientnet-{variante}",
                    "lado_imagem": lado,
                    "lote": lote,
                    "estado": estado,
                    "pico_alocado_mib": alocada,
                    "pico_reservado_mib": reservada,
                    "gpu_total_mib": total,
                }
            )
            print(linhas[-1], flush=True)
            if estado != "ok":
                break
    SAIDA.parent.mkdir(parents=False, exist_ok=False)
    with SAIDA.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)
    print("CSV:", SAIDA)


if __name__ == "__main__":
    main()
