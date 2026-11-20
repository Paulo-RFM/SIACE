#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


TILE_PATTERN = re.compile(r"^OrtoRP-3cm_(\d{3})_(\d{3})$")
SPLITS = ("train", "val", "test")
TARGETS = {"train": 0.60, "val": 0.20, "test": 0.20}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--masks-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=2000)
    parser.add_argument("--k-scan", action="store_true")
    parser.add_argument("--map", action="store_true", default=True)
    return parser.parse_args()


def tile_coordinates(tile):
    match = TILE_PATTERN.fullmatch(tile)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def read_tiles(images_dir):
    tiles = []
    ignored = []
    for path in Path(images_dir).iterdir():
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            coordinates = tile_coordinates(path.stem)
            if coordinates is None:
                ignored.append(path.name)
            else:
                tiles.append(path.stem)
    if not tiles:
        raise FileNotFoundError(f"Nenhum tile valido encontrado em {images_dir}")
    print(f"Tiles validos: {len(tiles)}; ignorados: {len(ignored)}")
    return sorted(tiles), ignored


class DisjointSet:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = Counter()

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.rank[ra] += self.rank[rb]


def connected_components(tiles, neighborhood=4):
    coordinate_to_tile = {tile_coordinates(tile): tile for tile in tiles}
    dsu = DisjointSet(tiles)
    for (row, col), tile in coordinate_to_tile.items():
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
        if neighborhood == 8:
            offsets += ((-1, -1), (-1, 1), (1, -1), (1, 1))
        for row_offset, col_offset in offsets:
            neighbor = coordinate_to_tile.get((row + row_offset, col + col_offset))
            if neighbor is not None:
                dsu.union(tile, neighbor)

    components = defaultdict(list)
    for tile in tiles:
        components[dsu.find(tile)].append(tile)
    return sorted(components.values(), key=lambda component: (-len(component), component[0]))


def allocate_components(components, total, seed, restarts):
    rng = random.Random(seed)
    best = None
    best_cost = float("inf")
    components_with_size = [
        {"tiles": component, "size": len(component)}
        for component in components
    ]

    for restart in range(restarts):
        ordered = components_with_size.copy()
        if restart:
            rng.shuffle(ordered)
            ordered.sort(key=lambda item: item["size"], reverse=True)
        counts = Counter()
        assignment = {}

        for component in ordered:
            options = list(SPLITS)
            if component["size"] >= total * 0.10:
                options = list(options)
            rng.shuffle(options)
            split = min(
                options,
                key=lambda candidate: sum(
                    (
                        (counts[candidate] + component["size"]) / total
                        - TARGETS[candidate]
                    ) ** 2
                    if split_name == candidate
                    else (counts[split_name] / total - TARGETS[split_name]) ** 2
                    for split_name in SPLITS
                ),
            )
            counts[split] += component["size"]
            assignment[frozenset(component["tiles"])] = split

        empty_splits = [split_name for split_name in SPLITS if counts[split_name] == 0]
        for empty_split in empty_splits:
            donor_split = max(SPLITS, key=lambda split_name: counts[split_name])
            donor_component = min(
                (
                    component for component in ordered
                    if assignment[frozenset(component["tiles"])] == donor_split
                    and counts[donor_split] - component["size"] > 0
                ),
                key=lambda component: abs(component["size"] - total / 5),
                default=None,
            )
            if donor_component:
                component_key = frozenset(donor_component["tiles"])
                counts[donor_split] -= donor_component["size"]
                counts[empty_split] += donor_component["size"]
                assignment[component_key] = empty_split

        cost = sum(
            (counts[split_name] / total - TARGETS[split_name]) ** 2
            for split_name in SPLITS
        )
        if cost < best_cost:
            best_cost = cost
            best = (counts.copy(), assignment.copy())
    return best, best_cost


def block_assignment(tiles, k, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for tile in tiles:
        row, col = tile_coordinates(tile)
        groups[f"r{(row - 1) // k + 1}_c{(col - 1) // k + 1}"].append(tile)
    names = sorted(groups)
    rng.shuffle(names)
    train_count = math.floor(len(names) * 0.60)
    val_count = math.floor(len(names) * 0.20)
    assignments = {}
    for index, group in enumerate(names):
        split = "train" if index < train_count else "val" if index < train_count + val_count else "test"
        for tile in groups[group]:
            assignments[tile] = (split, group)
    return assignments, len(groups)


def cross_neighbor_edges(mapping):
    split_by_tile = {row["tile"]: row["split"] for row in mapping}
    coordinate_to_tile = {
        tile_coordinates(row["tile"]): row["tile"]
        for row in mapping
    }
    edges = []
    seen = set()
    for row in mapping:
        row_number, col_number = tile_coordinates(row["tile"])
        for neighbor_coordinate in (
            (row_number - 1, col_number),
            (row_number + 1, col_number),
            (row_number, col_number - 1),
            (row_number, col_number + 1),
        ):
            neighbor = coordinate_to_tile.get(neighbor_coordinate)
            if neighbor is None:
                continue
            pair = tuple(sorted((row["tile"], neighbor)))
            if pair in seen:
                continue
            seen.add(pair)
            if split_by_tile[pair[0]] != split_by_tile[pair[1]]:
                edges.append(pair)
    return edges


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_map(output, mapping):
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        print(f"Mapa PNG nao gerado: Pillow indisponivel ({error})")
        return

    colors = {"train": (45, 114, 210), "val": (240, 167, 24), "test": (210, 45, 45)}
    scale = 18
    coordinates = [tile_coordinates(row["tile"]) for row in mapping]
    max_row = max(row for row, _ in coordinates)
    max_col = max(col for _, col in coordinates)
    image = Image.new("RGB", (max_col * scale, max_row * scale), "white")
    draw = ImageDraw.Draw(image)
    by_tile = {row["tile"]: row["split"] for row in mapping}
    for tile, split in by_tile.items():
        coordinates = tile_coordinates(tile)
        if coordinates is None:
            continue
        row_number, col_number = coordinates
        x0 = (col_number - 1) * scale
        y0 = (row_number - 1) * scale
        draw.rectangle((x0, y0, x0 + scale - 1, y0 + scale - 1), fill=colors[split])
        draw.rectangle((x0, y0, x0 + scale - 1, y0 + scale - 1), outline=(220, 220, 220))
    image.save(output, compress_level=9)


def write_k_chart(output, rows):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Grafico de vazamento nao gerado: Pillow indisponivel")
        return
    width, height = 1200, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 50, width - 50, height - 90
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    draw.line((left, top, left, bottom), fill="black", width=2)
    maximum = max([row["cross_edges_pct"] for row in rows] + [0.001])
    points = []
    for index, row in enumerate(rows):
        x = left + index * (right - left) / max(len(rows) - 1, 1)
        y = bottom - row["cross_edges_pct"] / maximum * (bottom - top)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=(210, 45, 45), width=4)
    for point, row in zip(points, rows):
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=(210, 45, 45))
        draw.text((point[0] - 10, bottom + 15), str(row["k"]), fill="black")
    draw.text((left + 20, 15), "Vazamento cruzado (%) por k", fill="black")
    draw.text((left, bottom + 35), "k de bloco espacial", fill="black")
    image.save(output)


def spatial_statistics(mapping):
    result = {}
    for split_name in SPLITS:
        rows = [
            tile_coordinates(row["tile"])[0]
            for row in mapping if row["split"] == split_name
        ]
        cols = [
            tile_coordinates(row["tile"])[1]
            for row in mapping if row["split"] == split_name
        ]
        if not rows or not cols:
            result[split_name] = {
                "count": len(rows),
                "row_range": None,
                "col_range": None,
                "row_std": None,
                "col_std": None,
            }
            continue
        mean_row = sum(rows) / len(rows)
        mean_col = sum(cols) / len(cols)
        result[split_name] = {
            "count": len(rows),
            "row_range": [min(rows), max(rows)],
            "col_range": [min(cols), max(cols)],
            "row_std": math.sqrt(sum((value - mean_row) ** 2 for value in rows) / len(rows)),
            "col_std": math.sqrt(sum((value - mean_col) ** 2 for value in cols) / len(cols)),
        }
    return result


def representativity_statistics(mapping, images_dir, masks_dir=None):
    try:
        from PIL import Image
        import numpy as np
    except ImportError as error:
        print(f"Representatividade nao verificavel: dependencia ausente ({error})")
        return None

    result = {}
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir) if masks_dir else None
    for split_name in SPLITS:
        total_mask_pixels = 0
        foreground_pixels = 0
        partial_tiles = 0
        tile_count = 0
        for row in mapping:
            if row["split"] != split_name:
                continue
            tile_count += 1
            image_path = images_dir / (row["tile"] + ".tif")
            with Image.open(image_path).convert("RGB") as image:
                image_array = __import__("numpy").asarray(image)
            partial_pixels = __import__("numpy").all(image_array == 0, axis=2)
            if partial_pixels.any():
                partial_tiles += 1

            mask_path = None
            if masks_dir:
                for suffix in (".png", ".tif", ".jpg"):
                    candidate = masks_dir / (row["tile"] + suffix)
                    if candidate.is_file():
                        mask_path = candidate
                        break
            if mask_path:
                with Image.open(mask_path).convert("RGB") as mask:
                    mask_array = __import__("numpy").asarray(mask)
                foreground = __import__("numpy").any(mask_array != 0, axis=2)
                total_mask_pixels += int(foreground.size)
                foreground_pixels += int(foreground.sum())
        result[split_name] = {
            "tiles": tile_count,
            "tiles_recorte_parcial": partial_tiles,
            "foreground_fraction": (
                foreground_pixels / total_mask_pixels if total_mask_pixels else None
            ),
        }
    return result


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tiles, ignored = read_tiles(args.images_dir)

    components_4 = connected_components(tiles, 4)
    components_8 = connected_components(tiles, 8)
    sizes_4 = sorted((len(component) for component in components_4), reverse=True)
    (counts, assignment), cost = allocate_components(
        components_4, len(tiles), args.seed, args.restarts
    )

    mapping = []
    for tile in tiles:
        split = next(
            split_name
            for component, split_name in assignment.items()
            if tile in component
        )
        mapping.append({"tile": tile, "split": split, "group": "component_" + str(
            sorted(components_4, key=lambda component: (-len(component), component[0])).index(
                next(component for component in components_4 if tile in component)
            )
        )})

    tiles_csv = output / "tiles_split.csv"
    write_csv(tiles_csv, ["tile", "split", "group"], mapping)
    cross_edges = cross_neighbor_edges(mapping)
    if cross_edges:
        raise RuntimeError(
            f"Vazamento residual inesperado apos split por componentes: {len(cross_edges)}"
        )

    component_counts = Counter(len(component) for component in components_4)
    manifest = {
        "method": "componentes_conexas",
        "neighborhood": 4,
        "seed": args.seed,
        "target": {"train": 0.60, "val": 0.20, "test": 0.20},
        "counts": dict(Counter(row["split"] for row in mapping)),
        "allocation_cost": cost,
        "component_count": len(components_4),
        "component_count_8": len(components_8),
        "largest_component": sizes_4[0] if sizes_4 else 0,
        "median_component": sizes_4[len(sizes_4) // 2] if sizes_4 else 0,
        "cross_edges": 0,
        "spatial_statistics": spatial_statistics(mapping),
        "ignored_files": ignored,
        "group_size_distribution": dict(sorted(component_counts.items())),
        "representativity": representativity_statistics(
            mapping, args.images_dir, args.masks_dir
        ),
        "tiles_by_split": {
            split_name: [row["tile"] for row in mapping if row["split"] == split_name]
            for split_name in SPLITS
        },
    }
    (output / "tiles_by_split.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "Split componentes: "
        + ", ".join(f"{name}={counts[name]}" for name in SPLITS)
        + f"; vazamento={len(cross_edges)}"
    )
    component_distribution = Counter(len(component) for component in components_4)
    write_csv(
        output / "component_sizes.csv",
        ["component_size", "component_count"],
        [
            {"component_size": size, "component_count": count}
            for size, count in sorted(component_distribution.items())
        ],
    )

    if args.map:
        write_map(output / "split_map.png", mapping)

    if args.k_scan:
        scan_rows = []
        for k in (5, 7, 9, 11, 13, 15, 19, 25):
            assignments, group_count = block_assignment(tiles, k, args.seed)
            block_mapping = [
                {"tile": tile, "split": assignments[tile][0], "group": assignments[tile][1]}
                for tile in tiles
            ]
            split_counts = Counter(row["split"] for row in block_mapping)
            deviation = sum(
                abs(split_counts[name] / len(block_mapping) - TARGETS[name])
                for name in SPLITS
            )
            cross = len(cross_neighbor_edges(block_mapping))
            cross_tiles = {
                tile
                for pair in cross_neighbor_edges(block_mapping)
                for tile in pair
            }
            scan_rows.append({
                "k": k,
                "groups": group_count,
                "train": split_counts["train"],
                "val": split_counts["val"],
                "test": split_counts["test"],
                "cross_edges": cross,
                "cross_edges_pct": cross / max(len(block_mapping), 1),
                "cross_tiles": len(cross_tiles),
                "cross_tiles_pct": len(cross_tiles) / len(block_mapping),
                "deviation_60_20_20": deviation,
            })
        write_csv(
            output / "k_scan.csv",
            ["k", "groups", "train", "val", "test", "cross_edges",
             "cross_edges_pct", "cross_tiles", "cross_tiles_pct",
             "deviation_60_20_20"],
            scan_rows,
        )
        write_k_chart(output / "k_scan_leakage.png", scan_rows)


if __name__ == "__main__":
    main()
