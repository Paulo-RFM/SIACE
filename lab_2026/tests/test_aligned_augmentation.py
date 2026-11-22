"""Testes do alinhamento imagem-mascara na augmentation (correcao de 2026-11-06).

Rodar a partir de lab_2026/: python -m unittest tests.test_aligned_augmentation
Limite configuravel: ALIGN_MIN_IOU (padrao 0,98) = IoU minimo aceito entre a mascara produzida e a
mascara de referencia derivada da propria imagem transformada.
"""
import os
import re
import unittest
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from aligned_augmentation import paired_random_resized_crop

LAB = Path(__file__).resolve().parents[1]
MIN_IOU = float(os.environ.get("ALIGN_MIN_IOU", "0.98"))
SCALES = [(0.85, 1.0), (0.4, 1.0), (0.3, 1.0), (0.7, 1.0)]  # faixas usadas nos scripts do benchmark


def synthetic_pair(seed=0, size=512):
    """Imagem com 'copas' (discos verdes) e mascara RGB derivada exatamente dela."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8) + np.array([90, 80, 70], np.uint8)
    yy, xx = np.mgrid[:size, :size]
    mask = np.zeros((size, size), bool)
    for _ in range(12):
        cy, cx, r = rng.integers(40, size - 40), rng.integers(40, size - 40), rng.integers(15, 60)
        mask |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    img[mask] = [30, 160, 40]
    mask_rgb = np.zeros((size, size, 3), np.uint8)
    mask_rgb[mask] = [255, 0, 0]
    return Image.fromarray(img), Image.fromarray(mask_rgb)


def binary(mask_pil):
    return np.any(np.array(mask_pil) != 0, axis=-1)


def from_image(img_pil):
    """Rotulo de referencia recalculado da imagem transformada (verde dominante = copa)."""
    a = np.array(img_pil).astype(int)
    return (a[..., 1] - a[..., 0]) > 60


def iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


class Alinhamento(unittest.TestCase):
    def test_mascara_alinhada_com_a_imagem(self):
        """Depois do crop, a mascara coincide com o rotulo recalculado da imagem transformada."""
        for k, scale in enumerate(SCALES):
            for s in range(10):
                img, mask = synthetic_pair(seed=100 * k + s)
                torch.manual_seed(s)
                im2, m2, params = paired_random_resized_crop(img, mask, (512, 512), scale)
                v = iou(binary(m2), from_image(im2))
                self.assertGreaterEqual(v, MIN_IOU, f"scale={scale} seed={s} params={params} IoU={v:.3f}")

    def test_mascara_igual_a_original_transformada(self):
        """A mascara produzida e exatamente a mascara original com o mesmo recorte (NEAREST)."""
        img, mask = synthetic_pair(seed=7)
        for s in range(10):
            torch.manual_seed(s)
            _, m2, (t, l, h, w) = paired_random_resized_crop(img, mask, (512, 512), (0.3, 1.0))
            ref = TF.resized_crop(mask, t, l, h, w, [512, 512], interpolation=InterpolationMode.NEAREST)
            self.assertEqual(iou(binary(m2), binary(ref)), 1.0)

    def test_mascara_sem_valores_intermediarios(self):
        """NEAREST: a mascara so tem as cores originais (sem bordas interpoladas que viram arvore)."""
        img, mask = synthetic_pair(seed=3)
        cores = {tuple(c) for c in np.array(mask).reshape(-1, 3)}
        for s in range(5):
            torch.manual_seed(s)
            _, m2, _ = paired_random_resized_crop(img, mask, (512, 512), (0.3, 1.0))
            self.assertTrue({tuple(c) for c in np.array(m2).reshape(-1, 3)} <= cores)

    def test_metodo_antigo_e_detectado(self):
        """Controle: o padrao antigo (random.seed + dois RandomResizedCrop) falha neste mesmo teste."""
        import random
        img, mask = synthetic_pair(seed=11)
        crop = transforms.RandomResizedCrop((512, 512), scale=(0.85, 1.0))
        piores = []
        for s in range(10):
            torch.manual_seed(s)
            random.seed(s); im2 = crop(img)
            random.seed(s); m2 = crop(mask)
            piores.append(iou(binary(m2), from_image(im2)))
        self.assertLess(min(piores), MIN_IOU)

    def test_tamanhos_diferentes_sao_recusados(self):
        img, mask = synthetic_pair()
        with self.assertRaises(ValueError):
            paired_random_resized_crop(img, mask.resize((256, 256)), (512, 512), (0.85, 1.0))


class Scripts(unittest.TestCase):
    TREINO = ["tree_segmentation_resnet.py", "tree_segmentation_resnet_26.py",
              "tree_segmentation_resnet_26_512semAug.py", "tree_segmentation_resnet_26_512semAug2.py",
              "tree_segmentation_resnet_26_512softAug2.py", "tree_segmentation_resnet_26_512softAug4.py",
              "tree_segmentation_resnet_26_512Aug5.py", "tree_segmentation_resnet_26_256softAug2.py",
              "banchmark_Unet.py", "Unet_plus_banchmark.py", "resnet3.py"]

    def test_nenhum_crop_separado_em_imagem_e_mascara(self):
        """Guarda estatica: nenhum script de treino aplica um transform aleatorio a mascara isoladamente."""
        for f in self.TREINO:
            src = (LAB / f).read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"mask\s*=\s*self\.(random_)?crop\(mask\)", src), f)
            self.assertIn("paired_random_resized_crop", src, f)


if __name__ == "__main__":
    unittest.main()
