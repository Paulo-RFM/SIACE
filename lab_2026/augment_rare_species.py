"""Balanceamento por aumento offline, aplicado SOMENTE ao treino do split espacial.

Versao historica (ate 2026-11-01): aumentava species_crops/species_labels.csv
inteiro ANTES do split, gerando species_crops_balanced/ com 1317 copias de
arvores de tiles val/test (vazamento). Ela foi substituida por esta versao; o
diretorio historico permanece intacto e nao deve ser usado para treino.

Cada copia gerada:
  * vem de uma arvore anotada do treino;
  * herda tile e particao (train) da arvore original;
  * recebe instance_uid proprio e parent_instance_uid = arvore original.

Uso:
  python augment_rare_species.py --manifest /home/paulomaciel/proj/species_dataset_v2/species_manifest.csv \
      --out-dir /home/paulomaciel/proj/species_dataset_v2/train_augmented --target 300
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance

import species_data as sd


def augment(img, rng):
    choice = rng.choice(["flip_h", "flip_v", "rotate", "color", "crop"])
    if choice == "flip_h":
        return img.transpose(Image.FLIP_LEFT_RIGHT), choice
    if choice == "flip_v":
        return img.transpose(Image.FLIP_TOP_BOTTOM), choice
    if choice == "rotate":
        angle = rng.choice([90, 180, 270])
        return img.rotate(angle, expand=False), f"rotate_{angle}"
    if choice == "color":
        f = rng.uniform(0.5, 1.5)
        return ImageEnhance.Color(img).enhance(f), f"color_{f:.3f}"
    w, h = img.size
    pad = int(min(w, h) * 0.05)
    return img.crop((pad, pad, w - pad, h - pad)).resize((w, h), Image.BILINEAR), "crop_5pct"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=str(sd.PROJECT_ROOT / "species_dataset_v2" / "species_manifest.csv"))
    p.add_argument("--split", default=None)
    p.add_argument("--out-dir", default=str(sd.PROJECT_ROOT / "species_dataset_v2" / "train_augmented"))
    p.add_argument("--target", type=int, default=None,
                   help="amostras alvo por classe no treino (padrao: tamanho da maior classe)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tile_split, split_info = sd.load_tile_split(sd.resolve_split_path(args.split))
    df = sd.validate_manifest(pd.read_csv(args.manifest), tile_split)
    base = df[df["origin"] == sd.ORIGIN_ANNOTATED]
    train = base[base["split"] == "train"]
    target = args.target or int(train["species"].value_counts().max())

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    if img_dir.exists() and any(img_dir.iterdir()):
        raise FileExistsError(f"{img_dir} ja contem arquivos; escolha outro --out-dir.")
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    new_rows = []
    for species, sub in sorted(train.groupby("species")):
        needed = max(0, target - len(sub))
        parents = sub.to_dict("records")
        for i in range(needed):
            parent = parents[i % len(parents)] if i < len(parents) else rng.choice(parents)
            img = Image.open(parent["crop_path"]).convert("RGB")
            aug_img, op = augment(img, rng)
            uid = f"{parent['instance_uid']}#aug{i:04d}"
            path = img_dir / (uid.replace("#", "__") + ".png")
            aug_img.save(path)
            new_rows.append({
                "crop_path": str(path), "tile": parent["tile"], "species": species,
                "origin": sd.ORIGIN_AUGMENTED, "instance_uid": uid,
                "parent_instance_uid": parent["instance_uid"], "split": "train",
                "augmentation_op": op, "mask_provenance": parent.get("mask_provenance"),
            })

    aug = pd.DataFrame(new_rows)
    combined = sd.validate_manifest(pd.concat([base, aug], ignore_index=True, sort=False), tile_split)
    aug.to_csv(out_dir / "train_augmented.csv", index=False)
    before = sd.counts_by_split(base)
    after = sd.counts_by_split(combined)
    report = {
        "manifest": args.manifest, "split_sha256": split_info["split_sha256"], "target_per_class": target,
        "seed": args.seed, "n_augmented": len(aug),
        "before": {str(k): {c: int(v) for c, v in r.items()} for k, r in before.iterrows()},
        "after": {str(k): {c: int(v) for c, v in r.items()} for k, r in after.iterrows()},
        "imbalance_ratio_before": {s: sd.imbalance_ratio(before[s]) for s in sd.VALID_SPLITS},
        "imbalance_ratio_after": {s: sd.imbalance_ratio(after[s]) for s in sd.VALID_SPLITS},
        "note": "val e test permanecem identicos (somente anotados).",
    }
    (out_dir / "augmentation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Antes:\n", before, "\nDepois:\n", after)
    print(json.dumps({k: report[k] for k in ("imbalance_ratio_before", "imbalance_ratio_after")}, indent=2))
    print("CSV:", out_dir / "train_augmented.csv")


if __name__ == "__main__":
    main()
