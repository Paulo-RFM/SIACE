#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


TRAINING_SCRIPTS = {
    "unet-basic": "tree_segmentation_resnet.py",
    "unet-512-semaug": "tree_segmentation_resnet_26_512semAug2.py",
    "unet-512-softaug-run1": "tree_segmentation_resnet_26_512softAug2.py",
    "unet-512-softaug-run2": "tree_segmentation_resnet_26_512softAug4.py",
    "unet-512-pseudo": "tree_segmentation_resnet_26_512Aug5.py",
    "unet-multiclass-bias": "resnet2.py",
    "unet-multiclass": "resnet3.py",
    "unet-encoder-benchmark": "banchmark_Unet.py",
    "unet-unetpp-benchmark": "Unet_plus_banchmark.py",
    "species-classification": "train_species_efficientnet.py",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Executa treinamentos U-Net/EfficientNet.")
    parser.add_argument(
        "experiments",
        nargs="*",
        choices=list(TRAINING_SCRIPTS),
        help="Experimentos a executar. Sem argumentos, executa todos.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Somente lista os comandos.")
    return parser.parse_args()


def main():
    args = parse_args()
    selected = args.experiments or list(TRAINING_SCRIPTS)
    base_dir = Path(__file__).resolve().parent

    print("Experimentos selecionados:")
    for name in selected:
        command = [sys.executable, str(base_dir / TRAINING_SCRIPTS[name])]
        print(f"- {name}: {' '.join(command)}")
        if args.dry_run:
            continue
        subprocess.run(command, cwd=base_dir, check=True)


if __name__ == "__main__":
    main()
