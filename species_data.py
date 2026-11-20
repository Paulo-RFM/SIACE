"""Regras de dados auditaveis para a classificacao de especies (SIACE).

Este modulo nao depende de PyTorch: concentra a leitura do split espacial,
a construcao/validacao do manifesto de recortes e as checagens bloqueantes
(vazamento, pseudorrotulos, aumentados). E usado por
build_species_manifest.py, augment_rare_species.py,
train_species_efficientnet.py e pelos testes em tests/.

Esquema canonico do manifesto (uma linha por recorte):
    crop_path            caminho absoluto do PNG
    tile                 tile de origem (stem, sem extensao)
    species              id inteiro da especie (codigo group_id do LabelMe)
    origin               anotado | aumentado | pseudorrotulo
    instance_uid         identificador da arvore original (tile#fonte#indice)
    parent_instance_uid  instancia original (igual a instance_uid se nao aumentado)
    split                train | val | test (herdado do tile)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/home/paulomaciel/proj")
DEFAULT_SPLIT_CANDIDATES = (
    PROJECT_ROOT / "unet_spatial_split" / "tiles_split.csv",
    PROJECT_ROOT / "lab_2026" / "unet_spatial_split" / "tiles_split.csv",
)
VALID_SPLITS = ("train", "val", "test")

ORIGIN_ANNOTATED = "anotado"
ORIGIN_AUGMENTED = "aumentado"
ORIGIN_PSEUDO = "pseudorrotulo"
VALID_ORIGINS = (ORIGIN_ANNOTATED, ORIGIN_AUGMENTED, ORIGIN_PSEUDO)
# Grafias historicas aceitas na leitura e normalizadas para o vocabulario acima.
ORIGIN_ALIASES = {
    "anotado": ORIGIN_ANNOTATED, "manual": ORIGIN_ANNOTATED, "annotated": ORIGIN_ANNOTATED,
    "aumentado": ORIGIN_AUGMENTED, "augmented": ORIGIN_AUGMENTED,
    "pseudorrotulo": ORIGIN_PSEUDO, "pseudo": ORIGIN_PSEUDO, "pseudolabel": ORIGIN_PSEUDO,
}

MANIFEST_COLUMNS = ["crop_path", "tile", "species", "origin", "instance_uid",
                    "parent_instance_uid", "split"]
# Aliases de colunas encontrados nos scripts historicos.
COLUMN_ALIASES = {
    "crop_path": ("crop_path", "crop"),
    "tile": ("tile", "source_tile", "image"),
    "species": ("species", "label"),
    "origin": ("origin", "source_origin"),
}
# Registro obrigatorio para qualquer pseudorrotulo (Parte 3C).
PSEUDO_REQUIRED_COLUMNS = (
    "generator_checkpoint", "generator_checkpoint_sha256", "confidence",
    "generation_threshold", "segmentation_checkpoint", "species_source",
)
# Valores de species_source que indicam rotulo vindo SO da auto-predicao.
AUTO_ONLY_SPECIES_SOURCES = {"auto_prediction", "auto", "model", "modelo", "predicao"}


class DataAuditError(ValueError):
    """Erro bloqueante de auditoria de dados."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text_lines(lines) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(str(line).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------
# Split espacial
# --------------------------------------------------------------------------

def resolve_split_path(explicit: str | Path | None = None) -> Path:
    """Resolve o tiles_split.csv real. Falha se nenhum caminho existir."""
    candidates = [Path(explicit)] if explicit else list(DEFAULT_SPLIT_CANDIDATES)
    for cand in candidates:
        cand = cand / "tiles_split.csv" if cand.is_dir() else cand
        if cand.is_file():
            return cand.resolve()
    raise DataAuditError(
        "Split espacial ausente. Procurado em: " + ", ".join(str(c) for c in candidates))


def load_tile_split(path: str | Path) -> tuple[dict, dict]:
    """Le e valida o split espacial. Retorna (tile->split, info de auditoria)."""
    path = Path(path)
    if not path.is_file():
        raise DataAuditError(f"Split espacial ausente: {path}")
    df = pd.read_csv(path)
    missing = {"tile", "split"} - set(df.columns)
    if missing:
        raise DataAuditError(f"{path} sem colunas obrigatorias: {sorted(missing)}")
    if df.empty:
        raise DataAuditError(f"{path} esta vazio.")
    df["tile"] = df["tile"].astype(str)
    df["split"] = df["split"].astype(str)
    bad = sorted(set(df["split"]) - set(VALID_SPLITS))
    if bad:
        raise DataAuditError(f"Particoes invalidas em {path}: {bad}")
    absent = [s for s in VALID_SPLITS if s not in set(df["split"])]
    if absent:
        raise DataAuditError(f"Split incompleto, particoes sem tiles: {absent}")
    multi = df.groupby("tile")["split"].nunique()
    multi = sorted(multi[multi > 1].index)
    if multi:
        raise DataAuditError(f"Tiles em mais de uma particao no split: {multi[:20]}")
    if "group" in df.columns:
        gmulti = df.groupby("group")["split"].nunique()
        gmulti = sorted(gmulti[gmulti > 1].index)
        if gmulti:
            raise DataAuditError(f"Componentes espaciais em mais de uma particao: {gmulti[:20]}")
    df = df.drop_duplicates("tile")
    info = {
        "split_path": str(path),
        "split_sha256": sha256_file(path),
        "tiles_per_split": {s: int((df["split"] == s).sum()) for s in VALID_SPLITS},
        "tiles_in_multiple_splits": 0,
    }
    return dict(zip(df["tile"], df["split"])), info


# --------------------------------------------------------------------------
# Normalizacao e validacao do manifesto
# --------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mapeia aliases historicos (crop, image, label, ...) para o esquema canonico.

    Falha se nenhuma das grafias aceitas de uma coluna obrigatoria existir.
    Nao inventa a coluna de origem: sem origin/source_origin o CSV e rejeitado.
    """
    out = df.copy()
    for canon, aliases in COLUMN_ALIASES.items():
        if canon in out.columns:
            continue
        found = [a for a in aliases if a in out.columns]
        if not found:
            raise DataAuditError(
                f"Coluna obrigatoria ausente: '{canon}' (aceitas: {list(aliases)}). "
                f"Colunas presentes: {list(df.columns)}")
        out[canon] = out[found[0]]
    out["tile"] = out["tile"].map(lambda v: Path(str(v)).stem)
    raw_origin = out["origin"].astype(str).str.strip().str.lower()
    unknown = sorted(set(raw_origin) - set(ORIGIN_ALIASES))
    if unknown:
        raise DataAuditError(f"Valores de origem desconhecidos: {unknown}")
    out["origin"] = raw_origin.map(ORIGIN_ALIASES)
    if "instance_uid" not in out.columns:
        raise DataAuditError("Coluna obrigatoria ausente: 'instance_uid' (arvore original).")
    if "parent_instance_uid" not in out.columns:
        out["parent_instance_uid"] = out["instance_uid"]
    out["parent_instance_uid"] = out["parent_instance_uid"].fillna(out["instance_uid"])
    out["species"] = pd.to_numeric(out["species"], errors="raise").astype(int)
    return out


def attach_split(df: pd.DataFrame, tile_split: dict) -> pd.DataFrame:
    """Cada recorte herda a particao do tile de origem; falha se o tile nao estiver no split."""
    out = df.copy()
    unknown = sorted(set(out["tile"]) - set(tile_split))
    if unknown:
        raise DataAuditError(f"Tiles de origem fora do split espacial: {unknown[:20]} "
                             f"(total {len(unknown)})")
    inherited = out["tile"].map(tile_split)
    if "split" in out.columns:
        conflict = out["split"].notna() & (out["split"].astype(str) != inherited)
        if conflict.any():
            raise DataAuditError(
                "Particao gravada diverge da particao do tile em "
                f"{int(conflict.sum())} linhas, ex.: {out.loc[conflict, 'crop_path'].head(3).tolist()}")
    out["split"] = inherited
    return out


def validate_pseudolabels(df: pd.DataFrame) -> None:
    pseudo = df[df["origin"] == ORIGIN_PSEUDO]
    if pseudo.empty:
        return
    outside = pseudo[pseudo["split"] != "train"]
    if not outside.empty:
        raise DataAuditError(
            f"Bloqueio: {len(outside)} pseudorrotulos fora do treino "
            f"({outside['split'].value_counts().to_dict()}).")
    missing = [c for c in PSEUDO_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataAuditError(f"Pseudorrotulos sem registro obrigatorio: {missing}")
    empty = [c for c in PSEUDO_REQUIRED_COLUMNS if pseudo[c].isna().any()
             or (pseudo[c].astype(str).str.strip() == "").any()]
    if empty:
        raise DataAuditError(f"Pseudorrotulos com campos de proveniencia vazios: {empty}")
    auto_only = pseudo["species_source"].astype(str).str.strip().str.lower().isin(AUTO_ONLY_SPECIES_SOURCES)
    if auto_only.any():
        raise DataAuditError(
            f"Bloqueio: {int(auto_only.sum())} pseudorrotulos cuja especie veio exclusivamente "
            "da auto-predicao (species_source). Exige revisao/verificacao independente.")


def validate_manifest(df: pd.DataFrame, tile_split: dict) -> pd.DataFrame:
    """Aplica todas as regras bloqueantes. Retorna o manifesto normalizado."""
    df = normalize_columns(df)
    df = attach_split(df, tile_split)

    # Nenhum tile em mais de uma particao (redundante com o split, mas verificado no manifesto).
    per_tile = df.groupby("tile")["split"].nunique()
    if (per_tile > 1).any():
        raise DataAuditError(f"Tile em varias particoes: {per_tile[per_tile > 1].index[:10].tolist()}")

    # Instancia original pertence a um unico tile e a uma unica particao.
    per_parent = df.groupby("parent_instance_uid")["split"].nunique()
    if (per_parent > 1).any():
        raise DataAuditError(
            "Derivados da mesma arvore em mais de uma particao: "
            f"{per_parent[per_parent > 1].index[:10].tolist()}")
    parent_tiles = df.groupby("parent_instance_uid")["tile"].nunique()
    if (parent_tiles > 1).any():
        raise DataAuditError("Instancia original associada a mais de um tile.")

    aug = df[df["origin"] == ORIGIN_AUGMENTED]
    if not aug.empty:
        bad = aug[aug["split"] != "train"]
        if not bad.empty:
            raise DataAuditError(f"Bloqueio: {len(bad)} amostras aumentadas fora do treino "
                                 f"({bad['split'].value_counts().to_dict()}).")
        roots = set(df.loc[df["origin"] != ORIGIN_AUGMENTED, "instance_uid"])
        orphan = aug[~aug["parent_instance_uid"].isin(roots)]
        if not orphan.empty:
            raise DataAuditError(f"{len(orphan)} aumentados sem instancia original no manifesto.")
        if (aug["parent_instance_uid"] == aug["instance_uid"]).any():
            raise DataAuditError("Aumentado com instance_uid igual ao pai; use um uid proprio.")

    validate_pseudolabels(df)

    for name in ("val", "test"):
        part = df[df["split"] == name]
        if (part["origin"] != ORIGIN_ANNOTATED).any():
            raise DataAuditError(
                f"Particao {name} contem origens nao anotadas: "
                f"{part['origin'].value_counts().to_dict()}")

    if df["crop_path"].duplicated().any():
        raise DataAuditError("crop_path duplicado no manifesto.")
    return df


# --------------------------------------------------------------------------
# Relatorios
# --------------------------------------------------------------------------

def counts_by_split(df: pd.DataFrame, origin: str | None = None) -> pd.DataFrame:
    sub = df if origin is None else df[df["origin"] == origin]
    tab = pd.crosstab(sub["species"], sub["split"]).reindex(columns=list(VALID_SPLITS), fill_value=0)
    return tab


def imbalance_ratio(counts: pd.Series) -> float | None:
    counts = counts[counts > 0]
    if counts.empty:
        return None
    return float(counts.max() / counts.min())


def test_eligibility(df: pd.DataFrame, min_samples: int = 10) -> dict:
    test = df[(df["split"] == "test") & (df["origin"] == ORIGIN_ANNOTATED)]
    counts = test["species"].value_counts().sort_index()
    all_species = sorted(df["species"].unique())
    per_species = {int(s): int(counts.get(s, 0)) for s in all_species}
    eligible = [s for s, n in per_species.items() if n >= min_samples]
    return {
        "min_samples": min_samples,
        "test_counts": per_species,
        "eligible_species": eligible,
        "below_min": [s for s, n in per_species.items() if n < min_samples],
        "n_eligible": len(eligible),
        "meaningful_test": len(eligible) > 0,
    }


def leakage_report(df: pd.DataFrame) -> dict:
    tiles = {s: set(df.loc[df["split"] == s, "tile"]) for s in VALID_SPLITS}
    parents = {s: set(df.loc[df["split"] == s, "parent_instance_uid"]) for s in VALID_SPLITS}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    return {
        "tiles_shared": {f"{a}&{b}": len(tiles[a] & tiles[b]) for a, b in pairs},
        "instances_shared": {f"{a}&{b}": len(parents[a] & parents[b]) for a, b in pairs},
        "tiles_per_split": {s: len(tiles[s]) for s in VALID_SPLITS},
    }


def load_species_names(path: str | Path | None, species_ids) -> tuple[dict, bool]:
    """Le mapping id->nome (JSON). Sem arquivo, usa 'especie_<id>' e sinaliza."""
    ids = sorted(int(s) for s in species_ids)
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        names = {int(k): str(v) for k, v in data.items()}
        missing = [i for i in ids if i not in names]
        if missing:
            raise DataAuditError(f"Mapping de nomes sem as especies {missing}")
        return {i: names[i] for i in ids}, True
    return {i: f"especie_{i}" for i in ids}, False


# --------------------------------------------------------------------------
# Paleta das mascaras multiclasse (make_mask_distin.py -> masks_distin*/)
# Verificada em 2026-11-02 contando as cores reais de lab_2026/masks_distin2 (1018
# arquivos) e masks_distin (707): exatamente estas 7 cores; (255,0,0) nao ocorre.
# --------------------------------------------------------------------------
SPECIES_PALETTE = {
    (0, 0, 0): 0,          # fundo
    (128, 128, 128): 1,    # arvore sem identificacao (tambem group_id ausente)
    (0, 255, 0): 2,        # palmeira
    (0, 0, 255): 3,        # arvore seca
    (255, 255, 0): 4,      # mangueira
    (0, 255, 255): 5,      # bananeira
    (255, 0, 255): 6,      # jabuticabeira
}


def rgb_mask_to_class(mask_rgb):
    """Converte mascara RGB (H, W, 3) em indices 0..6; falha se houver cor fora da paleta."""
    import numpy as np
    arr = np.asarray(mask_rgb, dtype=np.uint32)
    code = (arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]
    out = np.full(code.shape, 255, dtype=np.uint8)
    for (r, g, b), cls in SPECIES_PALETTE.items():
        out[code == ((r << 16) | (g << 8) | b)] = cls
    if (out == 255).any():
        raise DataAuditError(f"{int((out == 255).sum())} pixels com cor fora da paleta de especies.")
    return out


def check_mask_palette(paths) -> list:
    """Retorna [(arquivo, cores_invalidas)] para mascaras com cores fora da paleta."""
    import numpy as np
    from PIL import Image
    valid = {(r << 16) | (g << 8) | b for (r, g, b) in SPECIES_PALETTE}
    bad = []
    for p in paths:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint32)
        codes = set(np.unique((arr[..., 0] << 16) | (arr[..., 1] << 8) | arr[..., 2]).tolist())
        extra = codes - valid
        if extra:
            bad.append((str(p), sorted((c >> 16, (c >> 8) & 255, c & 255) for c in extra)[:5]))
    return bad
