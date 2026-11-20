# SIACE: detecção e classificação de árvores urbanas por imagens de VANT

Código e artefatos do Trabalho de Conclusão de Curso de Paulo Maciel (UFV – Campus Rio Paranaíba).
O sistema detecta copas de árvores em um ortomosaico RGB de VANT (pixel de ~3 cm) da área urbana de
Rio Paranaíba – MG, individualiza cada copa, classifica a espécie e georreferencia o resultado.

**Sistema final**

| Etapa | Método | Run |
|---|---|---|
| Segmentação binária (árvore × fundo) | SegFormer MiT-B1, limiar 0,40 congelado na validação | `runs/arq_segformer_mit_b1_20261110` |
| Individualização das copas | Watershed com marcadores h-maxima (h = 0,75 m, área mínima 3000 px) | `runs/individualizacao_val_segformer_20261111` |
| Classificação da espécie | EfficientNet-B0 | `runs/classif_20261102_163825_efficientnet_anotados` |
| Aplicação à cidade | 13.491 árvores | `runs/pipeline_cidade_segformer_20261111` |
| Envio de teste ao OpenStreetMap | 4.362 pontos `natural=tree`, servidor de testes | `runs/osm_envio_20261111_145947` |

Modelo experimentado e comparado: U-Net ResNet-34 (`runs/alinhado_20261105_2215_softAug4`), além de
DeepLabV3+ e U-Net++ no comparativo de arquiteturas.

## Protocolo de avaliação

- Divisão espacial dos *tiles* anotados em treino, validação e teste (`unet_spatial_split/tiles_split.csv`).
- Toda escolha (limiares, parâmetros, modelo) foi feita na validação; o teste foi avaliado uma única vez.
- Intervalos de confiança de 95% por *bootstrap* por *tile* (B = 2000, semente 42).
- Critérios do filtro de altura registrados antes dos resultados (`runs/filtro_altura_20261118/criterio.md`).

## Estrutura

```
*.py                 scripts (ficam na raiz: os caminhos internos usam ./runs)
cadeias/             scripts shell que encadearam os treinos e avaliações finais
runs/                artefatos dos runs usados na monografia (configs, logs, métricas, GeoJSON/CSV)
unet_spatial_split/  divisão espacial dos tiles em treino/validação/teste
perimetros/          perímetros KML usados nos mapas (urbano, UFV, centro)
modelos/             pesos finais (.pth) — não vão para o git, ver abaixo
tests/               testes (pytest)
```

### Ordem de execução

1. `make_mask.py`: máscaras binárias a partir dos polígonos LabelMe.
2. `build_unet_spatial_split.py`: divisão espacial (`unet_spatial_protocol.py` contém as regras).
3. Segmentação:
   - `cadeias/arquiteturas_20261110.sh`: treina SegFormer MiT-B1, DeepLabV3+ e U-Net++ com
     `tree_segmentation_arquiteturas.py` e compara com `avaliar_arquiteturas.py`;
   - `cadeias/alinhamento_retreino_20261105_2215.sh`: U-Net ResNet-34 (`tree_segmentation_resnet_26_512softAug4.py`;
     só a configuração `softAug4` é usada na monografia);
   - `cadeias/seeds_segformer_20261110.sh` e `cadeias/seeds_alinhamento_20261106.sh`: repetição com sementes;
   - `bootstrap_ic_segformer.py`: intervalos de confiança.
4. Classificação: `build_species_manifest.py` → `run_classification_suite.py` / `train_species_efficientnet.py`
   → `evaluate_multiclass_test.py`.
5. Ponta a ponta: `cadeias/refazer_segformer_20261111.sh` e `cadeias/refazer_segformer_hmax_20261111.sh`
   (calibração da individualização na validação, avaliação no teste com `evaluate_pipeline_test.py`,
   limiares do OSM com `osm_calibrar_limiares.py` e aplicação à cidade com `pipeline_cidade.py`).
6. OSM: `osm_envio_dev.py` (sem `--enviar` apenas simula). O token OAuth 2 do servidor de testes é lido da
   variável de ambiente `OSM_DEV_TOKEN` e não está no repositório.
7. Filtro de altura (não adotado): `filtro_altura.py` e `osm_filtro_altura.py`.
8. Figuras e mapas: `gerar_figuras_resultados.py`, `gerar_curvas_p1.py`, `gerar_fluxograma.py`,
   `gerar_mapa_cidade.py`, `gerar_mapa_urbano.py`.

## Dados não incluídos

O ortomosaico, os *tiles* (512 × 512 px, GeoTIFF), as anotações e as máscaras não estão no repositório pelo
tamanho. Os scripts e as cadeias têm os caminhos da máquina de origem (`/home/paulomaciel/proj/imagens`,
`/home/paulomaciel/proj/masks`, `/home/paulomaciel/Datasets/drone_rp_v01/tiles512`, ...); ajuste-os para o
seu ambiente.

Também ficaram de fora dos runs: checkpoints (`*.pth`), máscaras previstas (`mascaras/`), partes por *tile*
(`partes/`), predições de teste em imagem e o `copas.geojson` da cidade (251 MB).

## Modelos

`modelos/` contém os pesos finais, ignorados pelo git por tamanho:

- `segformer_mit_b1_best.pth` (55 MB)
- `unet_resnet34_best.pth` (98 MB)
- `efficientnet_b0_especies_best.pth` (16 MB)

Publique-os como anexos de uma *Release* do GitHub. Para usar, copie cada um para `checkpoint_best.pth` do run
correspondente em `runs/`.

## Ambiente

Python 3.10, PyTorch 2.5.1 (CUDA 12.1), segmentation-models-pytorch 0.5.0. Ver `requirements.txt`.
