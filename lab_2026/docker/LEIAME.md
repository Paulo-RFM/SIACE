# SIACE em contêiner Docker

O contêiner recebe um ortomosaico RGB em GeoTIFF e devolve os mesmos produtos do sistema:

- `arvores.geojson`: um ponto por árvore, com área da copa e espécie prevista;
- `copas.geojson`: o contorno de cada copa;
- os arquivos CSV, `summary.json` e `provenance.json`.

O fluxo é o da monografia, sem nenhuma mudança:

1. A imagem é reprojetada para EPSG:4326 numa grade de 3,19 cm.
2. É recortada em tiles de 512 × 512 px, sem sobreposição e com bordas brancas.
3. A SegFormer MiT-B1 segmenta as copas, o Watershed *h-maxima* separa as árvores e a EfficientNet-B0 indica a espécie.
4. Os pontos e as copas são georreferenciados.

## 1. Clonar só o necessário

O repositório tem cerca de 600 MB, a maior parte em resultados de execuções. O contêiner precisa de menos de 1 MB de código, mais os modelos. Por isso, use um clone parcial:

```bash
git clone --depth 1 --filter=blob:none --sparse -b siace-publicacao https://github.com/Paulo-RFM/SIACE.git siace
cd siace
git sparse-checkout set --no-cone lab_2026/docker/ lab_2026/.dockerignore \
  lab_2026/pipeline_cidade.py lab_2026/tree_georef.py lab_2026/separate_tree_crowns_whatershed.py \
  lab_2026/species_data.py \
  lab_2026/runs/arq_segformer_mit_b1_20261110/frozen_threshold.json \
  lab_2026/runs/res_segformer_8cm_20261121/frozen_threshold.json \
  lab_2026/runs/classif_20261102_163825_efficientnet_anotados/config.json \
  lab_2026/runs/individualizacao_val_segformer_20261111/frozen_instance_params_hmax.json \
  lab_2026/runs/arq_segformer_mit_b1_20261110/checkpoint_best.pth \
  lab_2026/runs/res_segformer_8cm_20261121/checkpoint_best.pth \
  lab_2026/runs/classif_20261102_163825_efficientnet_anotados/checkpoint_best.pth
cd lab_2026
```

## 2. Conferir os modelos

Os pesos (`.pth`, cerca de 125 MB no total) vêm no próprio clone do passo 1:

- SegFormer de 3 cm (o modelo do sistema);
- SegFormer de 8 cm;
- EfficientNet-B0, sem a qual o pipeline não roda.

O mesmo script serve para conferir o SHA-256 de cada um. Se faltar algum arquivo, ele tenta baixá-lo de uma release do GitHub:

```bash
bash docker/baixar_modelos.sh            # confere (e, se faltar, baixa da release "modelos-v1")
bash docker/baixar_modelos.sh <URL_BASE> # outra origem para o download
```

## 3. Construir e rodar

```bash
docker build -f docker/Dockerfile -t siace .
docker run --rm --gpus all -v /caminho/dos/dados:/dados siace processar /dados/orto.tif /dados/saida
docker run --rm --gpus all -v /caminho/dos/dados:/dados siace processar /dados/orto.tif /dados/saida --modelo 8cm
```

Opções:

| Opção | Efeito |
|---|---|
| `--modelo 3cm` (padrão) | Usa o modelo do sistema, treinado com o ortomosaico de 3,19 cm. |
| `--modelo 8cm` | Usa a variante treinada com imagens degradadas para 8 cm (IoU de teste 0,772). Indicada para imagens de 8 cm. |
| `--utm EPSG` | Define a projeção usada no cálculo de áreas. Sem a opção, a zona é escolhida pelo centro da imagem: SIRGAS 2000 no Brasil, WGS 84 fora dele. |
| `siace pipeline <tiles> <saida>` | Pula o preparo, para tiles que já estão na grade de 3,19 cm em EPSG:4326. |

**GPU:** para usar a GPU, o computador hospedeiro precisa de driver NVIDIA ≥ 525 e do `nvidia-container-toolkit`. Sem `--gpus all`, o contêiner roda em CPU, várias vezes mais lento.

**Tempo:** com a GPU GTX 1050 Ti, foram cerca de 9 minutos de inferência por km² de área imageada (João Pessoa, imagem de 8 cm).

**Disco:** os tiles ocupam cerca de 1,2 GB por km².

## Limitações (as mesmas da monografia)

- **Espécies:** foram treinadas só com árvores de Rio Paranaíba. Em outra cidade, elas não têm validade.
- **Árvores fundidas:** copas que se tocam tendem a ser contadas como uma árvore só, e matas contínuas viram polígonos grandes.
- **Falsos positivos:** há detecções falsas em mato baixo, arbustos e água. Em João Pessoa, a precisão de detecção, conferida visualmente em 100 árvores, foi de 92,9%.
- **Variante de 8 cm:** os parâmetros do *h-maxima* foram calibrados com o modelo de 3 cm e não foram recalibrados para ela.

## Estado

- **Fluxo testado fora do Docker:** em 22/11/2026, com a mesma estrutura de `/app` e um recorte de 2.500 × 2.500 px de João Pessoa, o fluxo gerou 169 tiles e 90 árvores com o modelo de 3 cm e 104 com o de 8 cm, e a zona UTM detectada foi 31985.
- **Imagem Docker nunca construída:** esta máquina não tem Docker. Em particular, falta verificar a instalação das dependências sobre a imagem base do GDAL.
