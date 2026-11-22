# SIACE: inventário de árvores urbanas a partir de ortomosaico RGB (inferência).
# Construir a partir de lab_2026/, depois de baixar os modelos (docker/baixar_modelos.sh):
#   docker build -f docker/Dockerfile -t siace .
# Rodar (GPU NVIDIA exige o nvidia-container-toolkit no hospedeiro; sem --gpus roda em CPU, bem mais lento):
#   docker run --rm --gpus all -v /caminho/dos/dados:/dados siace processar /dados/orto.tif /dados/saida
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.6.2
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
WORKDIR /app
# código do pipeline
COPY pipeline_cidade.py tree_georef.py separate_tree_crowns_whatershed.py species_data.py ./
COPY docker/slicer.py docker/preparar_ortomosaico.py docker/siace.sh ./
# modelos e parâmetros congelados (os .pth vêm de docker/baixar_modelos.sh; os .json estão no git)
COPY runs/arq_segformer_mit_b1_20261110/checkpoint_best.pth runs/arq_segformer_mit_b1_20261110/frozen_threshold.json runs/arq_segformer_mit_b1_20261110/
COPY runs/res_segformer_8cm_20261121/checkpoint_best.pth runs/res_segformer_8cm_20261121/frozen_threshold.json runs/res_segformer_8cm_20261121/
COPY runs/classif_20261102_163825_efficientnet_anotados/checkpoint_best.pth runs/classif_20261102_163825_efficientnet_anotados/config.json runs/classif_20261102_163825_efficientnet_anotados/
COPY runs/individualizacao_val_segformer_20261111/frozen_instance_params_hmax.json runs/individualizacao_val_segformer_20261111/
RUN chmod +x siace.sh && python3 -c "import torch, segmentation_models_pytorch, efficientnet_pytorch, cv2, skimage, shapely, pyproj; from osgeo import gdal_array; print('ok')"
ENTRYPOINT ["/app/siace.sh"]
CMD ["ajuda"]
