#!/usr/bin/env python3
import argparse
import csv
import re
import random
from collections import Counter, defaultdict
from pathlib import Path


TILE_PATTERN = re.compile(r"^OrtoRP-3cm_(\d{3})_(\d{3})$")
SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conta vizinhos espaciais imediatos no split persistido 60/20/20."
    )
    parser.add_argument(
        "split_csv",
        nargs="?",
        help="Arquivo tiles_split.csv persistido. Opcional quando --images-dir e usado.",
    )
    parser.add_argument(
        "--images-dir",
        help="Diretorio com os TIFFs. Se o CSV nao existir, gera um split temporal 60/20/20.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        help="Diretorio de saida opcional para os CSVs detalhados.",
        default=None,
    )
    return parser.parse_args()


def tile_coordinates(tile):
    match = TILE_PATTERN.fullmatch(tile)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def count_neighbors(mapping):
    split_by_tile = {row["tile"]: row["split"] for row in mapping}
    coordinates_by_tile = {
        row["tile"]: tile_coordinates(row["tile"])
        for row in mapping
    }
    coordinates_by_tile = {
        tile: coordinates
        for tile, coordinates in coordinates_by_tile.items()
        if coordinates is not None
    }
    coordinate_to_tile = {
        coordinates: tile
        for tile, coordinates in coordinates_by_tile.items()
    }

    tile_neighbors = defaultdict(list)
    edges = []
    seen_pairs = set()
    for tile, (row, col) in coordinates_by_tile.items():
        for neighbor_coordinate in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            neighbor = coordinate_to_tile.get(neighbor_coordinate)
            if neighbor is None:
                continue
            pair = tuple(sorted((tile, neighbor)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            tile_neighbors[tile].append(neighbor)
            tile_neighbors[neighbor].append(tile)
            edges.append(
                {
                    "tile_a": pair[0],
                    "tile_b": pair[1],
                    "split_a": split_by_tile[pair[0]],
                    "split_b": split_by_tile[pair[1]],
                }
            )

    return edges, tile_neighbors


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.images_dir:
        images_dir = Path(args.images_dir).expanduser()
        generated_csv = None
        if args.split_csv:
            csv_path = Path(args.split_csv).expanduser()
        else:
            csv_path = images_dir / "tiles_split_neighbors.csv"
            generated_csv = csv_path

        if generated_csv is not None and not csv_path.exists():
            tile_paths = sorted(
                {
                    path for path in images_dir.iterdir()
                    if path.is_file() and path.suffix.lower() == ".tif"
                },
                key=lambda path: path.name,
            )
            valid_tile_paths = [
                path for path in tile_paths
                if tile_coordinates(path.stem) is not None
            ]
            ignored_tile_paths = [
                path for path in tile_paths
                if tile_coordinates(path.stem) is None
            ]
            tiles = [path.stem for path in valid_tile_paths]
            if not tiles:
                raise FileNotFoundError(f"Nenhum TIFF encontrado em {images_dir}")
            if ignored_tile_paths:
                print(
                    "Arquivos fora do padrao ignorados: "
                    f"{len(ignored_tile_paths)}"
                )

            groups = defaultdict(list)
            for tile in tiles:
                row, col = tile_coordinates(tile)
                group = f"r{(row - 1) // 5 + 1:02d}_c{(col - 1) // 5 + 1:02d}"
                groups[group].append(tile)

            group_names = sorted(groups)
            random.Random(args.seed).shuffle(group_names)
            train_group_count = round(len(group_names) * 0.60)
            val_group_count = round(len(group_names) * 0.20)
            assignments = {}
            for group in group_names[:train_group_count]:
                assignments[group] = "train"
            for group in group_names[train_group_count:train_group_count + val_group_count]:
                assignments[group] = "val"
            for group in group_names[train_group_count + val_group_count:]:
                assignments[group] = "test"

            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["tile", "split", "group"])
                writer.writeheader()
                for tile in tiles:
                    row, col = tile_coordinates(tile)
                    group = f"r{(row - 1) // 5 + 1:02d}_c{(col - 1) // 5 + 1:02d}"
                    writer.writerow({"tile": tile, "split": assignments[group], "group": group})
            print(f"Split temporario gerado em: {csv_path}")
    elif args.split_csv:
        csv_path = Path(args.split_csv).expanduser()
    else:
        raise ValueError("Informe tiles_split.csv ou --images-dir.")
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        mapping = list(reader)
    if not reader.fieldnames:
        raise ValueError(f"Arquivo vazio: {csv_path}")
    required_columns = {"tile", "split"}
    if missing := required_columns - set(reader.fieldnames):
        raise ValueError(f"Colunas ausentes em {csv_path}: {sorted(missing)}")
    tiles = [row["tile"].strip() for row in mapping]
    if len(tiles) != len(set(tiles)):
        raise ValueError("O arquivo contem nomes de tile duplicados.")
    invalid_tiles = [
        row["tile"] for row in mapping
        if tile_coordinates(row["tile"]) is None
    ]
    if invalid_tiles:
        print(f"Tiles fora do padrao ignorados: {len(invalid_tiles)}")
        mapping = [
            row for row in mapping
            if tile_coordinates(row["tile"]) is not None
        ]

    for row in mapping:
        row["tile"] = row["tile"].strip()
        row["split"] = row["split"].strip().lower()
    invalid_splits = sorted(
        {row["split"] for row in mapping} - set(SPLITS)
    )
    if invalid_splits:
        raise ValueError(f"Particoes invalidas: {invalid_splits}")

    counts_by_split = Counter(row["split"] for row in mapping)
    total = len(mapping)
    edges, tile_neighbors = count_neighbors(mapping)

    internal_edges = Counter()
    cross_edges = Counter()
    cross_endpoint_counts = Counter()
    internal_endpoint_counts = defaultdict(set)

    for edge in edges:
        split_a = edge["split_a"]
        split_b = edge["split_b"]
        if split_a == split_b:
            internal_edges[split_a] += 1
            internal_endpoint_counts[split_a].add(edge["tile_a"])
            internal_endpoint_counts[split_a].add(edge["tile_b"])
        else:
            pair = tuple(sorted((split_a, split_b)))
            cross_edges[pair] += 1
            cross_endpoint_counts[edge["tile_a"]] += 1
            cross_endpoint_counts[edge["tile_b"]] += 1

    print("=== Distribuicao do split 60/20/20 ===")
    for split_name in SPLITS:
        count = int(counts_by_split.get(split_name, 0))
        print(f"{split_name:5s}: {count:4d} tiles ({count / total * 100:5.1f}%)")
    print(f"total : {total:4d} tiles")
    print(f"\nArestas de vizinhanca presentes: {len(edges)}")

    print("\n=== Vizinhanca interna por particao ===")
    for split_name in SPLITS:
        print(
            f"{split_name:5s}: {internal_edges[split_name]} pares de vizinhos "
            f"envolvendo {len(internal_endpoint_counts[split_name])} tiles"
        )

    print("\n=== Vazamento espacial entre particoes ===")
    cross_total = sum(cross_edges.values())
    print(f"total de pares entre splits: {cross_total}")
    for split_a in SPLITS:
        for split_b in SPLITS:
            if split_a >= split_b:
                continue
            pair = tuple(sorted((split_a, split_b)))
            print(f"{split_a} <-> {split_b}: {cross_edges[pair]} pares")
    print(
        f"Tiles com pelo menos um vizinho em split diferente: "
        f"{len(cross_endpoint_counts)} de {total} "
        f"({len(cross_endpoint_counts) / total * 100:.2f}%)"
    )
    for split_name in SPLITS:
        count = sum(
            1
            for row in mapping if row["split"] == split_name
            for tile in (row["tile"],)
            if any(
                edge["split_a"] != edge["split_b"]
                and (edge["tile_a"] == tile or edge["tile_b"] == tile)
                for edge in edges
            )
        )
        print(f"{split_name}: {count} tiles com vizinho em outra particao")

    if mapping and "group" in mapping[0]:
        split_groups = defaultdict(set)
        for row in mapping:
            split_groups[row["group"]].add(row["split"])
        overlapping = {
            group: splits
            for group, splits in split_groups.items()
            if len(splits) > 1
        }
        if len(overlapping):
            raise ValueError(
                f"Grupos espaciais aparecem em mais de uma particao: "
                f"{sorted(overlapping)}"
            )
        print("\nIntegridade espacial: OK; nenhum grupo aparece em mais de uma particao.")

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        write_csv(
            output_dir / "spatial_neighbors.csv",
            list(edges[0].keys()) + ["same_split"] if edges else
            ["tile_a", "tile_b", "split_a", "split_b", "same_split"],
            [
                {
                    **edge,
                    "same_split": edge["split_a"] == edge["split_b"],
                }
                for edge in edges
            ],
        )

        tile_report_fields = list(mapping[0].keys()) + [
            "neighbor_count",
            "cross_split_neighbor_count",
        ]
        write_csv(
            output_dir / "tiles_neighbor_counts.csv",
            tile_report_fields,
            [
                {
                    **row,
                    "neighbor_count": len(tile_neighbors[row["tile"]]),
                    "cross_split_neighbor_count": cross_endpoint_counts[row["tile"]],
                }
                for row in mapping
            ],
        )
        print(f"\nRelatorios escritos em: {output_dir}")


if __name__ == "__main__":
    main()
