# Calibração do watershed com máscaras de REFERÊNCIA (2026-11-02)

Objetivo: medir a separação de copas isolada do erro da segmentação (limite superior).
Máscaras binárias de referência: `/home/paulomaciel/proj/masks/` (make_mask.py).
Verdade: polígonos `tree` de `labels/` e `labels2/` (incluindo os sem `group_id`).
Casamento: um-para-um (algoritmo húngaro), TP se IoU >= 0,5. Script: `eval_watershed_vs_polygons.py`.

1. Busca de parâmetros SOMENTE na validação (203 tiles): `val_sweep.csv`.
2. Critério declarado antes da busca: maior F1 de detecção, que deu peak_footprint=251, min_area=1000 (`frozen_watershed_params.json`).
3. Teste (204 tiles) avaliado uma única vez com os parâmetros congelados:

| Métrica (teste, máscara de referência) | Default (fp=15, min_area=150) | Congelado (fp=251, min_area=1000) |
|---|---|---|
| Árvores anotadas / instâncias | 418 / 804 | 418 / 360 |
| Precisão | 0,461 | 0,928 |
| Recall | 0,888 | 0,799 |
| F1 | 0,607 | 0,859 |
| IoU médio dos acertos | 0,846 | 0,938 |
| Erro médio de contagem por tile | +1,89 | −0,28 |
| MAE de contagem por tile | 1,91 | 0,44 |
| Tiles com contagem exata | 35,3% | 69,6% |

A linha "Default" do teste foi medida antes da calibração, como diagnóstico.
O número que vale para o sistema real é o fim a fim: watershed sobre a máscara PREDITA pelo P1 (limiar 0,65) nos mesmos 204 tiles de teste (pendente; precisa de GPU).

## Georreferenciamento (teste da conversão, `tree_georef.py`)

`georef_test_referencia/` traz as copas do watershed sobre as máscaras de REFERÊNCIA dos 204 tiles de teste,
convertidas para WGS 84: `copas.geojson` (polígonos), `arvores.geojson` (centroides), `arvores.csv` (lat/lon, área em m²).
360 instâncias por tile resultaram em 273 árvores após unir 57 copas cortadas pela borda entre tiles.
Serve para validar a conversão (abrir no QGIS sobre os tiles). NÃO é o produto final, que usará a máscara predita pelo P1.

Correções em `tree_georef.py` (2026-11-02, ~18h30), refletidas na pasta acima:
- contorno simplificado com tolerância de 1 px: GeoJSON de 6,9 MB para 0,5 MB, área mediana alterada em 0,25%;
- copas unidas entre tiles: a fresta (~1 cm) da emenda é costurada (buffer de ±0,10 m). Antes, a união virava MultiPolygon e só a maior parte era mantida, com área média de 64% da real. Agora a razão área/pixels é 0,994, igual à das copas de um tile;
- centroide de copa unida = centro de massa das partes ponderado por pixels (antes: centroide geométrico ou representative_point, instável). 273/273 centroides dentro da copa.
