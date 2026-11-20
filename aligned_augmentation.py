"""Augmentation geometrica sincronizada entre imagem e mascara (correcao de 2026-11-06).

Problema corrigido: os scripts de segmentacao chamavam `random.seed(seed)` antes de aplicar
`transforms.RandomResizedCrop` na imagem e de novo antes de aplicar na mascara, supondo que o mesmo
seed gerasse o mesmo recorte. Mas o torchvision sorteia os parametros do recorte com o gerador do
PyTorch (`torch.randint`/`torch.empty().uniform_()`), e nao com o modulo `random` do Python. Por isso
imagem e mascara recebiam recortes diferentes. Medido em 200 mascaras reais do treino com escala
0,85-1,0: deslocamento mediano de 18 px, IoU mediano de 0,80 entre a mascara certa e a usada.
Alem disso, a mascara era reamostrada com interpolacao bilinear, o padrao do RandomResizedCrop; como
qualquer pixel nao preto vira arvore, isso dilatava a borda da copa.

Solucao: sortear os parametros UMA vez e aplicar as mesmas coordenadas aos dois, com BILINEAR na
imagem e NEAREST na mascara. Os flips ja eram sincronizados (uma decisao aplicada aos dois) e
continuam iguais.

Log das transformacoes: com a variavel de ambiente ALIGNED_AUG_LOG=<arquivo.csv>, cada recorte
aplicado acrescenta uma linha "pid,top,left,height,width" ao arquivo.
"""
import os

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, RandomResizedCrop


def paired_random_resized_crop(img, mask, size, scale, ratio=(3.0 / 4.0, 4.0 / 3.0), generator=None):
    """Recorta imagem e mascara (PIL) com os MESMOS parametros e redimensiona para `size`.

    Imagem: BILINEAR. Mascara: NEAREST (rotulos sem valores intermediarios).
    Retorna (img, mask, params), params = (top, left, height, width), para log e testes.
    """
    if img.size != mask.size:
        raise ValueError(f"imagem {img.size} e mascara {mask.size} com tamanhos diferentes")
    size = list(size) if isinstance(size, (tuple, list)) else [size, size]
    if generator is None:
        top, left, h, w = RandomResizedCrop.get_params(img, list(scale), list(ratio))
    else:  # caminho deterministico usado nos testes
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
            top, left, h, w = RandomResizedCrop.get_params(img, list(scale), list(ratio))
    img_out = TF.resized_crop(img, top, left, h, w, size, interpolation=InterpolationMode.BILINEAR)
    mask_out = TF.resized_crop(mask, top, left, h, w, size, interpolation=InterpolationMode.NEAREST)
    params = (int(top), int(left), int(h), int(w))
    log = os.environ.get("ALIGNED_AUG_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()},{params[0]},{params[1]},{params[2]},{params[3]}\n")
    return img_out, mask_out, params
