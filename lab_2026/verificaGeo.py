import os
import rasterio

imagem = "/home/paulomaciel/proj/pseudo_dataset/images/OrtoRP-3cm_209_138.tif"

# 1. Verifica se o arquivo existe no disco
if not os.path.isfile(imagem):
    print("Arquivo não encontrado:", imagem)
else:
    print("Arquivo encontrado. Tentando abrir...")

    # 2. Tenta abrir com rasterio
    try:
        with rasterio.open(imagem) as ds:
            print("Arquivo carregado com sucesso!")
            print("Formato:", ds.driver)
            print("Dimensões (linhas x colunas):", ds.height, "x", ds.width)
            print("Número de bandas:", ds.count)
            print("CRS:", ds.crs)
            print("Resolução:", ds.res)
    except rasterio.errors.RasterioIOError as e:
        print("Erro ao abrir o arquivo com rasterio:")
        print(e)
