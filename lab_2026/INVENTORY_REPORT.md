# INVENTORY_REPORT — Recortes de espécies (SIACE)

Data: 2026-11-02. Gerado por `build_species_manifest.py` e verificado por
`train_species_efficientnet.py --audit-only`.

## Fontes e hashes

| Item | Caminho | SHA-256 |
|---|---|---|
| Split espacial | `/home/paulomaciel/proj/unet_spatial_split/tiles_split.csv` | `8adf3e724ef313ad7ed6260d181f87fc77ecefad9c16b9c5de2ba5607838b658` (o mesmo do run P1) |
| Manifesto novo (2ª versão, com `group_id` ausente → classe 1) | `/home/paulomaciel/proj/species_dataset_v2/species_manifest.csv` | `48b3b88310f742baad5e8b6313c0a7c7daeb99e424f8be3b93c6319800d44948` |
| Lista de teste (caminhos ordenados) | `inventory.json → test_hash` | `eb8f635085059391e0dd208b261fe64a62252a4575238d15c9bd725207db2712` |
| Inventário completo | `/home/paulomaciel/proj/species_dataset_v2/inventory.json` | — |

Split: 1018 tiles, sendo 611 de treino, 203 de validação e 204 de teste, em 246 componentes conexas (vizinhança-4).
Nenhum tile ou componente aparece em mais de uma partição.

## Inventário de recortes

| Conjunto | Local | Linhas | Origem | Usável no treino novo? |
|---|---|---|---|---|
| Manuais, `labels/` | `species_crops/` + `species_labels.csv` | 1530 | Polígonos LabelMe (`generate_crops_from_json.py`); 25/25 recortes regenerados idênticos byte a byte | **Sim**, reutilizados |
| Manuais, `labels2/` | `species_crops_v2/` (novo) | 571 (565 + 6 sem `group_id`) | Mesmos polígonos e mesmo algoritmo; nunca tinham entrado no CSV de espécies | **Sim** |
| Aumentados históricos | `species_crops_balanced/` | 5200 linhas (3670 cópias) | Aumento feito **antes** do split; 1317 cópias derivam de árvores de tiles de val/test | **Não**, por vazamento |
| Pseudorrótulos `pseudo_full.csv` | `dataset_auto/` | 1233 | O classificador histórico re-rotulou os **próprios recortes manuais** (concordância de 95,9% com o rótulo manual) | **Não** |
| Pseudorrótulos `pseudo_balanced.csv` | `dataset_auto/` | 344 | Tiles de `/home/paulomaciel/Datasets/drone_rp_v01/tiles512` (341 fora do split); confiança ≥ 0,95 | **Não** |
| Recortes de watershed | `masks_instances/` | 0 | Diretório vazio | — |

Regras de rótulo na construção do manifesto:
- 18 polígonos `tree` sem `group_id` (12 em `labels/`, 6 em `labels2/`) recebem a **classe 1, árvore sem identificação**. A decisão é do autor (2026-11-02) e segue a mesma regra de `make_mask_distin.py`. Os recortes foram gerados em `species_crops_v2/`, e a regra fica registrada em `species_rule = group_id_ausente->1`.
- 1 polígono com menos de 3 pontos é descartado.
- O arquivo `labels2/OrtoRP-3cm_082_013 - Copia.json` inteiro. O `imagePath` dele também aponta para "- Copia", então o tile real não pode ser confirmado.
- 1 rótulo com código 11 foi corrigido para 1, conforme `fix_species.py`. A correção está registrada em `species_raw`.

Manifesto novo: **2113 recortes anotados** de 1015 tiles (1542 de `labels/`, 571 de `labels2/`). Não há aumentados nem pseudorrótulos.

## Contagens por espécie e partição (somente anotados)

| Espécie (código) | Treino | Validação | Teste | Total |
|---|---|---|---|---|
| 1 árvore sem identificação | 1012 | 308 | 322 | 1642 |
| 2 palmeira | 128 | 20 | 20 | 168 |
| 3 árvore seca | 22 | 6 | **3** | 31 |
| 4 mangueira | 28 | 21 | 19 | 68 |
| 5 bananeira | 94 | 38 | 48 | 180 |
| 6 jabuticabeira | 12 | 6 | **6** | 24 |
| **Total** | 1296 | 399 | 418 | 2113 |

Distribuição só com `labels/`, o conjunto histórico: teste = {1: 268, 2: 17, 3: **0**, 4: 19, 5: 48, 6: 6}.
Sem `labels2/`, a espécie 3 não teria nenhuma amostra de teste.

Razão entre a maior e a menor classe: treino 84,3; validação 51,3; teste 107,3.

## Tiles por partição

| Partição | Tiles no split | Tiles com ≥1 recorte | Recortes por tile (mín / mediana / máx) |
|---|---|---|---|
| train | 611 | 610 | 1 / 2 / 7 |
| val | 203 | 203 | 1 / 2 / 6 |
| test | 204 | 202 | 1 / 1,5 / 11 |

Tiles distintos por espécie:

| Espécie | Treino | Validação | Teste |
|---|---|---|---|
| 1 | 567 | 187 | 183 |
| 2 | 92 | 19 | 17 |
| 3 | 21 | 5 | 3 |
| 4 | 26 | 18 | 16 |
| 5 | 67 | 25 | 31 |
| 6 | 12 | 6 | 6 |

## Vazamento (manifesto novo)

| Verificação | Resultado |
|---|---|
| Tiles compartilhados train&val / train&test / val&test | 0 / 0 / 0 |
| Árvores (instance_uid) compartilhadas | 0 / 0 / 0 |
| Aumentados fora do treino | 0 (não há aumentados) |
| Pseudorrótulos | 0 |
| Pares de tiles de partições diferentes com aresta comum | 0 |
| Pares de tiles de partições diferentes que se tocam só na diagonal | **53**: limitação do critério de vizinhança-4 do split da segmentação, mantido sem alteração |

## Classes elegíveis ao teste (≥ 10 amostras)

- **Elegíveis (4):** espécies 1, 2, 4 e 5.
- **Abaixo do mínimo (aviso):** espécie 3 (3 amostras) e espécie 6 (6 amostras).
- Há pelo menos uma classe com ≥ 10 amostras, então o teste é considerado significativo para treinar. As métricas por classe das espécies 3 e 6 não são estáveis.
