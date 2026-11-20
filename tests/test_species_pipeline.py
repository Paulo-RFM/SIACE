"""Testes de consistencia do pipeline de especies (unittest, sem GPU).

Executar:  python -m unittest discover -s tests -v   (a partir de lab_2026/)
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import species_data as sd  # noqa: E402
import species_metrics as sm  # noqa: E402

TILE_SPLIT = {"t_train": "train", "t_train2": "train", "t_val": "val", "t_test": "test"}


def row(tile, uid, origin="anotado", species=1, parent=None, **extra):
    r = {"crop_path": f"/x/{uid}.png", "tile": tile, "species": species, "origin": origin,
         "instance_uid": uid, "parent_instance_uid": parent or uid}
    r.update(extra)
    return r


def base_rows():
    return [row("t_train", "a1"), row("t_train2", "a2", species=2),
            row("t_val", "v1"), row("t_test", "s1"), row("t_test", "s2", species=2)]


class SplitTests(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        f.write(text); f.close()
        return f.name

    def test_split_absent(self):
        with self.assertRaises(sd.DataAuditError):
            sd.load_tile_split("/nao/existe/tiles_split.csv")
        with self.assertRaises(sd.DataAuditError):
            sd.resolve_split_path("/nao/existe")

    def test_split_incomplete(self):
        path = self.write("tile,split\na,train\nb,val\n")
        with self.assertRaises(sd.DataAuditError):
            sd.load_tile_split(path)

    def test_tile_in_multiple_partitions(self):
        path = self.write("tile,split\na,train\na,val\nb,test\nc,val\n")
        with self.assertRaises(sd.DataAuditError):
            sd.load_tile_split(path)

    def test_split_columns(self):
        path = self.write("tile,particao\na,train\n")
        with self.assertRaises(sd.DataAuditError):
            sd.load_tile_split(path)

    def test_real_split_is_valid(self):
        path = sd.resolve_split_path()
        mapping, info = sd.load_tile_split(path)
        self.assertEqual(info["tiles_in_multiple_splits"], 0)
        self.assertEqual(sum(info["tiles_per_split"].values()), len(mapping))


class ManifestTests(unittest.TestCase):
    def test_valid_manifest_passes(self):
        df = sd.validate_manifest(pd.DataFrame(base_rows()), TILE_SPLIT)
        self.assertEqual(set(df["split"]), {"train", "val", "test"})

    def test_missing_origin_column(self):
        df = pd.DataFrame(base_rows()).drop(columns=["origin"])
        with self.assertRaises(sd.DataAuditError):
            sd.validate_manifest(df, TILE_SPLIT)

    def test_missing_expected_columns(self):
        for col in ("crop_path", "tile", "species", "instance_uid"):
            df = pd.DataFrame(base_rows()).drop(columns=[col])
            with self.subTest(col=col), self.assertRaises(sd.DataAuditError):
                sd.validate_manifest(df, TILE_SPLIT)

    def test_column_aliases_accepted(self):
        df = pd.DataFrame(base_rows()).rename(columns={"crop_path": "crop", "tile": "image",
                                                       "species": "label", "origin": "source_origin"})
        out = sd.validate_manifest(df, TILE_SPLIT)
        self.assertIn("crop_path", out.columns)

    def test_tile_not_in_split(self):
        df = pd.DataFrame(base_rows() + [row("t_desconhecido", "z1")])
        with self.assertRaises(sd.DataAuditError):
            sd.validate_manifest(df, TILE_SPLIT)

    def test_recorded_split_conflict(self):
        rows = base_rows(); rows[0]["split"] = "test"
        with self.assertRaises(sd.DataAuditError):
            sd.validate_manifest(pd.DataFrame(rows), TILE_SPLIT)

    def test_augmented_in_test(self):
        df = pd.DataFrame(base_rows() + [row("t_test", "s1#aug0", "aumentado", parent="s1")])
        with self.assertRaises(sd.DataAuditError):
            sd.validate_manifest(df, TILE_SPLIT)

    def test_augmented_in_val(self):
        df = pd.DataFrame(base_rows() + [row("t_val", "v1#aug0", "aumentado", parent="v1")])
        with self.assertRaises(sd.DataAuditError):
            sd.validate_manifest(df, TILE_SPLIT)

    def test_augmented_train_ok(self):
        df = pd.DataFrame(base_rows() + [row("t_train", "a1#aug0", "aumentado", parent="a1")])
        out = sd.validate_manifest(df, TILE_SPLIT)
        self.assertEqual((out["origin"] == "aumentado").sum(), 1)

    def test_augmented_parent_in_other_partition(self):
        # copia registrada num tile de treino mas derivada de arvore do teste
        df = pd.DataFrame(base_rows() + [row("t_train", "s1#aug0", "aumentado", parent="s1")])
        with self.assertRaises(sd.DataAuditError):
            sd.validate_manifest(df, TILE_SPLIT)

    def test_augmented_orphan(self):
        df = pd.DataFrame(base_rows() + [row("t_train", "x#aug0", "aumentado", parent="x")])
        with self.assertRaises(sd.DataAuditError):
            sd.validate_manifest(df, TILE_SPLIT)


PSEUDO_OK = dict(generator_checkpoint="/c.pth", generator_checkpoint_sha256="abc", confidence=0.9,
                 generation_threshold=0.5, segmentation_checkpoint="/s.pth", species_source="revisado_manual")


class PseudoTests(unittest.TestCase):
    def test_pseudo_outside_train(self):
        for tile in ("t_val", "t_test"):
            df = pd.DataFrame(base_rows() + [row(tile, "p1", "pseudorrotulo", **PSEUDO_OK)])
            with self.subTest(tile=tile), self.assertRaises(sd.DataAuditError):
                sd.validate_manifest(df, TILE_SPLIT)

    def test_pseudo_missing_provenance(self):
        extra = dict(PSEUDO_OK); extra.pop("generator_checkpoint_sha256")
        df = pd.DataFrame(base_rows() + [row("t_train", "p1", "pseudorrotulo", **extra)])
        with self.assertRaises(sd.DataAuditError):
            sd.validate_manifest(df, TILE_SPLIT)

    def test_pseudo_auto_only_species(self):
        extra = dict(PSEUDO_OK, species_source="auto_prediction")
        df = pd.DataFrame(base_rows() + [row("t_train", "p1", "pseudorrotulo", **extra)])
        with self.assertRaises(sd.DataAuditError):
            sd.validate_manifest(df, TILE_SPLIT)

    def test_pseudo_train_ok(self):
        df = pd.DataFrame(base_rows() + [row("t_train", "p1", "pseudorrotulo", **PSEUDO_OK)])
        out = sd.validate_manifest(df, TILE_SPLIT)
        self.assertEqual((out["origin"] == "pseudorrotulo").sum(), 1)

    def test_historical_pseudo_csv_rejected(self):
        path = sd.PROJECT_ROOT / "dataset_auto" / "pseudo_full.csv"
        if not path.is_file():
            self.skipTest("pseudo_full.csv indisponivel")
        with self.assertRaises(sd.DataAuditError):
            sd.validate_manifest(pd.read_csv(path), TILE_SPLIT)

    def test_historical_balanced_csv_rejected(self):
        path = sd.PROJECT_ROOT / "species_crops_balanced" / "species_labels_balanced.csv"
        if not path.is_file():
            self.skipTest("CSV balanceado historico indisponivel")
        with self.assertRaises(sd.DataAuditError):
            sd.validate_manifest(pd.read_csv(path), TILE_SPLIT)


class EligibilityTests(unittest.TestCase):
    def test_no_species_with_10(self):
        df = sd.validate_manifest(pd.DataFrame(base_rows()), TILE_SPLIT)
        elig = sd.test_eligibility(df, 10)
        self.assertFalse(elig["meaningful_test"])
        self.assertEqual(sorted(elig["below_min"]), [1, 2])


class MetricsTests(unittest.TestCase):
    def test_metrics_shapes(self):
        y = np.array([0, 0, 1, 1, 2, 2]); p = np.array([0, 1, 1, 1, 2, 0])
        probs = np.eye(3)[p] * 0.8 + 0.2 / 3
        g = sm.global_metrics(y, p, [0, 1, 2])
        self.assertAlmostEqual(g["accuracy"], 4 / 6)
        pc = sm.per_class_metrics(y, p, probs, [0, 1, 2], ["a", "b", "c"], [1, 2, 3])
        self.assertEqual(list(pc["support"]), [2, 2, 2])
        self.assertAlmostEqual(pc.loc[0, "specificity"], 3 / 4)
        _, ece = sm.calibration(y, probs)
        self.assertGreaterEqual(ece, 0.0)

    def test_absent_class_no_auc(self):
        y = np.array([0, 0, 1]); p = np.array([0, 1, 1]); probs = np.eye(3)[p]
        pc = sm.per_class_metrics(y, p, probs, [0, 1, 2], ["a", "b", "c"], [1, 2, 3])
        self.assertTrue(pd.isna(pc.loc[2, "auc_ovr"]))


if __name__ == "__main__":
    unittest.main()


class PaletteTests(unittest.TestCase):
    def test_palette_mapping(self):
        mask = np.array([[[0, 0, 0], [128, 128, 128], [0, 255, 0], [0, 0, 255]],
                         [[255, 255, 0], [0, 255, 255], [255, 0, 255], [0, 0, 0]]], dtype=np.uint8)
        self.assertEqual(sd.rgb_mask_to_class(mask).tolist(), [[0, 1, 2, 3], [4, 5, 6, 0]])

    def test_unknown_color_rejected(self):
        mask = np.array([[[255, 0, 0]]], dtype=np.uint8)  # cor antiga dos scripts, inexistente nas mascaras
        with self.assertRaises(sd.DataAuditError):
            sd.rgb_mask_to_class(mask)

    def test_scripts_use_real_palette(self):
        import re
        base = Path(__file__).resolve().parents[1]
        for script in ("resnet2.py", "resnet3.py"):
            src = (base / script).read_text(encoding="utf-8")
            block = re.search(r"color_to_class = (\{.*?\n\})", src, re.S).group(1)
            self.assertEqual(eval(block), sd.SPECIES_PALETTE, script)


class SuiteTests(unittest.TestCase):
    def test_disabled_and_unknown_experiments(self):
        import run_classification_suite as suite
        self.assertEqual(suite.select_experiments(["all"]), list(suite.EXPERIMENTS))
        with self.assertRaises(ValueError):
            suite.select_experiments(["catboost_final.py"])
        with self.assertRaises(ValueError):
            suite.select_experiments(["nao_existe"])

    def test_species_names_file(self):
        names, from_file = sd.load_species_names(Path(__file__).resolve().parents[1] / "species_names.json",
                                                 range(1, 7))
        self.assertTrue(from_file)
        self.assertEqual(names[2], "palmeira")
