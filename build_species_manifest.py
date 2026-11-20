"""Constroi o manifesto auditavel de recortes de especies a partir das anotacoes manuais.

Fonte dos rotulos: poligonos LabelMe (label == "tree", group_id = especie) em
labels/ e labels2/. O recorte e a bounding box do poligono com PADDING pixels,
redimensionada para CROP_SIZE x CROP_SIZE, mesmo algoritmo de
generate_crops_from_json.py (que gerou species_crops/species_labels.csv).

Nenhum recorte depende de mascara predita: segmentation_checkpoint = None e
mask_provenance = "manual_polygon_labelme".

Recortes ja existentes em species_crops/ sao reaproveitados (verificacao por
regeneracao de amostra); recortes novos vao para um diretorio separado
(--new-crops-dir), sem sobrescrever nada.

Uso:
    python build_species_manifest.py --out-dir /home/paulomaciel/proj/species_dataset_v2
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import species_data as sd

CROP_SIZE = 512
PADDING = 20
# Correcao historica de fix_species.py: codigo 11 digitado por engano para a especie 1.
SPECIES_CODE_FIXES = {11: 1}
# Poligono de arvore sem group_id = arvore sem identificacao (classe 1), decisao do
# autor em 2026-11-02; e a mesma regra de make_mask_distin.py (None -> cinza = classe 1).
MISSING_GROUP_ID_CLASS = 1
# JSONs excluidos explicitamente (motivo registrado no manifesto).
DEFAULT_EXCLUDED_JSON = {
    "OrtoRP-3cm_082_013 - Copia.json":
        "Arquivo '- Copia' sem imagem correspondente; imagePath tambem aponta para '- Copia'. "
        "Tile real nao confirmado pelos metadados.",
}


def polygon_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def make_crop(img: np.ndarray, points) -> np.ndarray:
    """Replica generate_crops_from_json.py (linhas 83-93)."""
    x0, y0, x1, y1 = polygon_bbox(points)
    x0 -= PADDING; y0 -= PADDING; x1 += PADDING; y1 += PADDING
    h, w = img.shape[:2]
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(w - 1, int(x1)); y1 = min(h - 1, int(y1))
    crop = img[y0:y1 + 1, x0:x1 + 1]
    return np.array(Image.fromarray(crop).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label-dirs", nargs="+",
                   default=[str(sd.PROJECT_ROOT / "labels"), str(sd.PROJECT_ROOT / "labels2")])
    p.add_argument("--images-dir", default=str(sd.PROJECT_ROOT / "imagens"))
    p.add_argument("--existing-crops-dir", default=str(sd.PROJECT_ROOT / "species_crops"))
    p.add_argument("--new-crops-dir", default=str(sd.PROJECT_ROOT / "species_crops_v2"))
    p.add_argument("--split", default=None, help="tiles_split.csv (resolvido automaticamente)")
    p.add_argument("--out-dir", default=str(sd.PROJECT_ROOT / "species_dataset_v2"))
    p.add_argument("--verify-sample", type=int, default=25,
                   help="recortes existentes regenerados para conferir equivalencia byte a byte")
    p.add_argument("--dry-run", action="store_true", help="nao grava recortes novos")
    return p.parse_args()


def main():
    args = parse_args()
    split_path = sd.resolve_split_path(args.split)
    tile_split, split_info = sd.load_tile_split(split_path)
    print(f"Split: {split_path}\n  sha256={split_info['split_sha256']}\n  tiles={split_info['tiles_per_split']}")

    rows, excluded, skipped = [], [], []
    for label_dir in args.label_dirs:
        label_dir = Path(label_dir)
        if not label_dir.is_dir():
            raise sd.DataAuditError(f"Diretorio de anotacoes ausente: {label_dir}")
        for jp in sorted(label_dir.glob("*.json")):
            if jp.name in DEFAULT_EXCLUDED_JSON:
                excluded.append({"json": str(jp), "reason": DEFAULT_EXCLUDED_JSON[jp.name]})
                continue
            data = json.loads(jp.read_text(encoding="utf-8"))
            tile = jp.stem
            for idx, shape in enumerate(data.get("shapes", []), start=1):
                reason = None
                if shape.get("label") != "tree":
                    reason = f"label={shape.get('label')!r}"
                elif len(shape.get("points", [])) < 3:
                    reason = "poligono com <3 pontos"
                if reason:
                    skipped.append({"json": str(jp), "shape_index": idx, "reason": reason})
                    continue
                raw = shape.get("group_id")
                if raw is None:
                    species, species_rule = MISSING_GROUP_ID_CLASS, "group_id_ausente->1"
                else:
                    raw = int(raw)
                    species = SPECIES_CODE_FIXES.get(raw, raw)
                    species_rule = "correcao_11->1" if raw in SPECIES_CODE_FIXES else "group_id"
                rows.append({
                    "tile": tile,
                    "annotation_json": str(jp),
                    "annotation_source": label_dir.name,
                    "shape_index": idx,
                    "species_raw": raw,
                    "species": species,
                    "species_rule": species_rule,
                    "points": shape["points"],
                })

    df = pd.DataFrame(rows)
    unknown = sorted(set(df["tile"]) - set(tile_split))
    if unknown:
        raise sd.DataAuditError(f"Anotacoes de tiles fora do split: {unknown}")
    dup = df.groupby("tile")["annotation_source"].nunique()
    if (dup > 1).any():
        raise sd.DataAuditError(f"Tile anotado em mais de um diretorio: {dup[dup > 1].index.tolist()}")

    existing_dir = Path(args.existing_crops_dir)
    new_dir = Path(args.new_crops_dir)
    if not args.dry_run:
        new_dir.mkdir(parents=True, exist_ok=True)

    # Verificacao: os recortes existentes sao reproduziveis pelo algoritmo documentado?
    existing_mask = df.apply(
        lambda r: (existing_dir / f"{r.tile}_tree_{r.shape_index:04d}.png").is_file(), axis=1)
    rng = random.Random(42)
    sample_idx = rng.sample(list(df.index[existing_mask]), min(args.verify_sample, int(existing_mask.sum())))
    verify_ok = 0
    img_cache: dict[str, np.ndarray] = {}

    def load_tile(tile):
        if tile not in img_cache:
            path = Path(args.images_dir) / f"{tile}.tif"
            if not path.is_file():
                raise sd.DataAuditError(f"Imagem do tile ausente: {path}")
            img_cache.clear()
            img_cache[tile] = np.array(Image.open(path).convert("RGB"))
        return img_cache[tile]

    for i in sample_idx:
        r = df.loc[i]
        regenerated = make_crop(load_tile(r.tile), r.points)
        stored = np.array(Image.open(existing_dir / f"{r.tile}_tree_{r.shape_index:04d}.png").convert("RGB"))
        verify_ok += int(stored.shape == regenerated.shape and np.array_equal(stored, regenerated))
    print(f"Verificacao de recortes existentes: {verify_ok}/{len(sample_idx)} identicos byte a byte")
    if sample_idx and verify_ok != len(sample_idx):
        raise sd.DataAuditError("Recortes existentes nao correspondem ao algoritmo documentado.")

    crop_paths, crop_status = [], []
    for i, r in df.sort_values(["tile", "shape_index"]).iterrows():
        name = f"{r.tile}_tree_{r.shape_index:04d}.png"
        if existing_mask.loc[i]:
            crop_paths.append((i, str(existing_dir / name))); crop_status.append((i, "reutilizado"))
            continue
        out = new_dir / name
        if out.exists():
            crop_status.append((i, "existente_v2"))
        elif not args.dry_run:
            Image.fromarray(make_crop(load_tile(r.tile), r.points)).save(out)
            crop_status.append((i, "gerado"))
        else:
            crop_status.append((i, "dry_run"))
        crop_paths.append((i, str(out)))
    df["crop_path"] = pd.Series(dict(crop_paths))
    df["crop_status"] = pd.Series(dict(crop_status))
    df["origin"] = sd.ORIGIN_ANNOTATED
    df["instance_uid"] = df.apply(lambda r: f"{r.tile}#{r.annotation_source}#{r.shape_index:04d}", axis=1)
    df["parent_instance_uid"] = df["instance_uid"]
    df["mask_provenance"] = "manual_polygon_labelme"
    df["segmentation_checkpoint"] = None
    if not args.dry_run:
        df["crop_sha256"] = df["crop_path"].map(sd.sha256_file)
    df = df.drop(columns=["points"])

    validated = sd.validate_manifest(df, tile_split)
    validated = validated.sort_values(["split", "tile", "shape_index"]).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "species_manifest.csv"
    validated.to_csv(manifest_path, index=False)
    pd.DataFrame(excluded).to_csv(out_dir / "excluded_annotation_files.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped_shapes.csv", index=False)

    counts = sd.counts_by_split(validated)
    leak = sd.leakage_report(validated)
    elig = sd.test_eligibility(validated)
    inventory = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "split": split_info,
        "manifest": str(manifest_path),
        "manifest_sha256": sd.sha256_file(manifest_path),
        "label_dirs": args.label_dirs,
        "crop_algorithm": {"source": "generate_crops_from_json.py", "padding_px": PADDING,
                           "crop_size": CROP_SIZE, "resize": "bilinear"},
        "existing_crops_verified": {"sample": len(sample_idx), "identical": verify_ok},
        "species_code_fixes": {str(k): v for k, v in SPECIES_CODE_FIXES.items()},
        "missing_group_id_class": MISSING_GROUP_ID_CLASS,
        "species_rule_counts": validated["species_rule"].value_counts().to_dict(),
        "mask_provenance": "manual_polygon_labelme",
        "segmentation_checkpoint": None,
        "n_crops": int(len(validated)),
        "n_by_source": validated["annotation_source"].value_counts().to_dict(),
        "crop_status": validated["crop_status"].value_counts().to_dict(),
        "counts_species_by_split": {str(k): {c: int(v) for c, v in row.items()} for k, row in counts.iterrows()},
        "imbalance_ratio": {s: sd.imbalance_ratio(counts[s]) for s in sd.VALID_SPLITS},
        "test_hash": sd.sha256_text_lines(sorted(validated.loc[validated.split == "test", "crop_path"])),
        "leakage": leak,
        "test_eligibility": elig,
        "excluded_annotation_files": excluded,
        "n_skipped_shapes": len(skipped),
        "skipped_reasons": pd.Series([s["reason"] for s in skipped]).value_counts().to_dict() if skipped else {},
    }
    (out_dir / "inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: inventory[k] for k in ("n_crops", "n_by_source", "crop_status", "leakage",
                                                 "imbalance_ratio", "test_eligibility")},
                     indent=2, ensure_ascii=False))
    print(counts.assign(total=counts.sum(axis=1)))
    print("Manifesto:", manifest_path)


if __name__ == "__main__":
    main()
