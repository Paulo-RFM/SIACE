"""Testes do georreferenciamento (tree_georef.py). Rodar a partir de lab_2026/:
    python -m unittest tests.test_georef
Os testes com tiles reais sao pulados se o diretorio de tiles nao existir.
"""
import unittest
from pathlib import Path

import numpy as np

import tree_georef as tg

CITY = Path("/home/paulomaciel/Datasets/drone_rp_v01/tiles512")


class PixelToLonLat(unittest.TestCase):
    GT = (-46.0, -19.0, 3e-7, 3e-7)

    def test_canto_superior_esquerdo(self):
        self.assertEqual(tg.pixel_to_lonlat(0, 0, self.GT), (-46.0, -19.0))

    def test_eixo_y_para_baixo_diminui_latitude(self):
        lon, lat = tg.pixel_to_lonlat(0, 512, self.GT)
        self.assertAlmostEqual(lat, -19.0 - 512 * 3e-7)
        self.assertEqual(lon, -46.0)

    def test_centroide_de_um_pixel_usa_meio_pixel(self):
        m = np.zeros((512, 512), np.int32)
        m[10, 20] = 1
        m[10:12, 20:22] = 1  # 2x2 pixels -> centro em (21, 11) nas bordas de pixel
        rec = tg.tile_records(m, "t", self.GT, m.shape)[0]
        lon, lat = tg.pixel_to_lonlat(21.0, 11.0, self.GT)
        self.assertAlmostEqual(rec["centroid"].x, lon, places=12)
        self.assertAlmostEqual(rec["centroid"].y, lat, places=12)


class UniaoNaBorda(unittest.TestCase):
    """Duas copas separadas no tile A tocam a mesma copa do tile B."""
    D = 1e-5  # ~1 m

    def df(self):
        import pandas as pd
        from shapely.geometry import box
        d, x0, y0 = self.D, -46.25, -19.18
        a1 = box(x0, y0, x0 + d, y0 + d)                    # tile A, encosta na borda x0+d
        a2 = box(x0, y0 + 2 * d, x0 + d, y0 + 3 * d)        # tile A, separada de a1
        b = box(x0 + d, y0, x0 + 2 * d, y0 + 3 * d)         # tile B, toca a1 (1 m) e a2 (1 m)
        recs = [("A", 1, a1), ("A", 2, a2), ("B", 1, b)]
        return pd.DataFrame([{"tile": t, "instance_id": i, "n_pixels": int(g.area / 1e-12), "touches_tile_border": True,
                              "geometry": g, "centroid": g.centroid} for t, i, g in recs])

    def test_encadeada_funde_tudo(self):
        self.assertEqual(len(tg.merge_border_crowns(self.df(), rule="encadeada")), 1)

    def test_uma_parte_por_tile_nao_refunde_copas_do_mesmo_tile(self):
        out = tg.merge_border_crowns(self.df(), rule="uma_parte_por_tile")
        self.assertEqual(len(out), 2)
        for parts in out["parts"]:
            tiles = [p.split("#")[0] for p in parts.split(";")]
            self.assertEqual(len(tiles), len(set(tiles)))


@unittest.skipUnless(CITY.is_dir(), "tiles da cidade indisponiveis")
class TilesReais(unittest.TestCase):
    def test_grade_continua_entre_tiles_vizinhos(self):
        a = tg.read_geotransform(CITY / "OrtoRP-3cm_140_120.tif")
        leste = tg.read_geotransform(CITY / "OrtoRP-3cm_140_121.tif")
        sul = tg.read_geotransform(CITY / "OrtoRP-3cm_141_120.tif")
        self.assertAlmostEqual(leste[0], a[0] + 512 * a[2], places=9)
        self.assertAlmostEqual(sul[1], a[1] - 512 * a[3], places=9)

    def test_tamanho_do_pixel_em_metros(self):
        lon0, lat0, sx, sy = tg.read_geotransform(CITY / "OrtoRP-3cm_140_120.tif")
        dx = sx * 111_320 * np.cos(np.radians(lat0))
        dy = sy * 110_574
        self.assertTrue(0.025 < dx < 0.04 and 0.025 < dy < 0.04, (dx, dy))


if __name__ == "__main__":
    unittest.main()
