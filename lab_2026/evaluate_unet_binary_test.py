#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from segmentation_models_pytorch import Unet
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from unet_spatial_protocol import frozen_threshold, hash_files, load_split, paired_tiles, valid_pixel_mask


class TestDataset(Dataset):
    def __init__(self, images, masks, size):
        self.images = images
        self.masks = masks
        self.size = size
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("RGB")
        image = image.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        valid = torch.from_numpy(valid_pixel_mask(image)).bool()
        target = torch.from_numpy(np.any(np.asarray(mask) != 0, axis=2)).float().unsqueeze(0)
        return self.to_tensor(image), target, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()

    threshold = frozen_threshold(args.threshold)
    pairs = paired_tiles(args.images, args.masks)
    image_paths = [image for image, _ in pairs]
    split, _, _ = load_split(args.split, image_paths)
    mask_by_image = {Path(image).stem: mask for image, mask in pairs}
    test_images = split["test"]
    test_masks = [mask_by_image[Path(image).stem] for image in test_images]

    dataset = TestDataset(test_images, test_masks, (args.img_size, args.img_size))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                        pin_memory=True, persistent_workers=True, drop_last=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    confusion = np.zeros((2, 2), dtype=np.int64)
    started = time.perf_counter()
    prediction_dir = Path(args.output) / "predictions_test"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    rng = random.Random(42)
    sample_count = min(20, len(test_images))
    selected_indices = set(rng.sample(range(len(test_images)), sample_count))
    with torch.no_grad():
        for index, (image, target, valid) in enumerate(loader):
            image = image.to(device)
            probability = torch.sigmoid(model(image))[0, 0].cpu().numpy()
            target_np = target[0, 0].numpy()
            valid_np = valid[0].numpy()
            predicted = (probability >= threshold).astype(np.uint8)
            valid_target = target_np[valid_np].astype(np.uint8)
            valid_predicted = predicted[valid_np].astype(np.uint8)
            confusion += np.bincount(
                valid_target * 2 + valid_predicted, minlength=4
            ).reshape(2, 2)

            Image.fromarray((predicted * 255).astype(np.uint8)).save(
                prediction_dir / f"{Path(test_images[index]).stem}.png"
            )
            if index in selected_indices:
                selected.append(index)
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(Image.open(test_images[index]).convert("RGB"))
                axes[0].set_title("Imagem")
                axes[1].imshow(target_np, cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("Mascara real")
                axes[2].imshow(predicted, cmap="gray", vmin=0, vmax=1)
                axes[2].set_title("Mascara prevista")
                for axis in axes:
                    axis.axis("off")
                fig.tight_layout()
                fig.savefig(prediction_dir / f"{Path(test_images[index]).stem}_panel_seed42.png", dpi=150)
                plt.close(fig)
    inference_s = time.perf_counter() - started

    tn, fp, fn, tp = confusion.ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    iou = tp / max(tp + fp + fn, 1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(confusion).to_csv(output / "test_confusion_matrix.csv", index=False, header=False)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(confusion, cmap="Blues")
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axis.text(column, row, str(confusion[row, column]), ha="center", va="center")
    axis.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    axis.set_yticks([0, 1], ["Real 0", "Real 1"])
    fig.tight_layout()
    fig.savefig(output / "test_confusion_matrix.png", dpi=150)
    plt.close(fig)
    positive_fraction_test = float(np.sum(confusion[1, :]) / confusion.sum())
    pd.DataFrame([{
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "positive_fraction_test": positive_fraction_test, "accuracy": (tp + tn) / max(confusion.sum(), 1),
        "precision": precision, "recall": recall, "f1": f1, "iou": iou,
        "inference_total_s": inference_s, "inference_per_tile_s": inference_s / len(test_images),
        "test_hash": hash_files(test_images),
    }]).to_csv(output / "test_metrics.csv", index=False)
    print(f"Teste avaliado uma unica vez: {len(test_images)} tiles, threshold={threshold:.4f}")


if __name__ == "__main__":
    main()
