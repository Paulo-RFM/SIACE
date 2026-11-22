"""Testes de osm_envio_dev.py sem acesso a rede. Rodar a partir de lab_2026/: python -m unittest tests.test_osm_envio"""
import unittest
import xml.etree.ElementTree as ET

import pandas as pd

import osm_envio_dev as oe

LIM = {"area_max_m2": 100.0, "limiar_seg_prob_mean": 0.8, "conf_especie": 0.7, "classes_com_especie": [4, 5]}
TAGS = {"confirmado_pelo_autor": True, "classes": {"4": {"tags": {"genus": "Mangifera"}}, "5": {"tags": {"genus": "Musa"}}}}


def trees():
    return pd.DataFrame([
        # tree_id, area, seg, classe, conf
        dict(tree_id=1, area_m2=20, seg_prob_mean=0.90, pred_class=4, confidence=0.95, lon=-46.25, lat=-19.18, pred_name="mangueira"),
        dict(tree_id=2, area_m2=20, seg_prob_mean=0.90, pred_class=4, confidence=0.60, lon=-46.25, lat=-19.18, pred_name="mangueira"),
        dict(tree_id=3, area_m2=20, seg_prob_mean=0.70, pred_class=5, confidence=0.99, lon=-46.25, lat=-19.18, pred_name="bananeira"),
        dict(tree_id=4, area_m2=900, seg_prob_mean=0.99, pred_class=1, confidence=0.99, lon=-46.25, lat=-19.18, pred_name="sem"),
        dict(tree_id=5, area_m2=20, seg_prob_mean=0.85, pred_class=2, confidence=0.99, lon=-46.25, lat=-19.18, pred_name="palmeira"),
    ])


class Selecao(unittest.TestCase):
    def test_filtros(self):
        s = oe.select(trees(), LIM, TAGS)
        self.assertEqual(list(s.tree_id), [1, 2, 5])            # 3: seg baixo; 4: macico
        self.assertEqual(list(s.com_especie), [True, False, False])  # 2: conf baixa; 5: classe nao liberada

    def test_bananeira_vai_como_planta_so_com_especie(self):
        t = trees()
        t.loc[t.tree_id == 3, "seg_prob_mean"] = 0.95
        t = pd.concat([t, t[t.tree_id == 3].assign(tree_id=6, confidence=0.50)])  # bananeira de baixa confianca
        s = oe.select(t, LIM, TAGS)
        self.assertIn(3, list(s.tree_id))
        self.assertNotIn(6, list(s.tree_id))
        tags = oe.node_tags(s[s.tree_id == 3].iloc[0], TAGS)
        self.assertEqual(tags, {"natural": "plant", "genus": "Musa"})
        sem = oe.select(t, LIM, {**TAGS, "confirmado_pelo_autor": False})
        self.assertNotIn(3, list(sem.tree_id))

    def test_sem_confirmacao_nao_indica_especie(self):
        s = oe.select(trees(), LIM, {**TAGS, "confirmado_pelo_autor": False})
        self.assertFalse(s.com_especie.any())

    def test_exige_seg_prob_mean(self):
        with self.assertRaises(SystemExit):
            oe.select(trees().drop(columns="seg_prob_mean"), LIM, TAGS)


class Xml(unittest.TestCase):
    def test_osmchange(self):
        s = oe.select(trees(), LIM, TAGS)
        root = ET.fromstring(oe.osc_xml(s, TAGS, changeset="42"))
        nodes = list(root.iter("node"))
        self.assertEqual([n.get("id") for n in nodes], ["-1", "-2", "-3"])
        self.assertTrue(all(n.get("changeset") == "42" for n in nodes))
        t0 = {t.get("k"): t.get("v") for t in nodes[0].iter("tag")}
        t1 = {t.get("k"): t.get("v") for t in nodes[1].iter("tag")}
        self.assertEqual(t0, {"natural": "tree", "genus": "Mangifera"})
        self.assertEqual(t1, {"natural": "tree"})

    def test_servidor_e_o_de_testes(self):
        self.assertTrue(oe.API.startswith("https://master.apis.dev.openstreetmap.org/"))


if __name__ == "__main__":
    unittest.main()
