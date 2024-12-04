import os
import numpy as np
from PIL import Image

# Diretório contendo os tiles de predição
output_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\predicts\44_resnet"

# Função para carregar as máscaras e suas coordenadas
def load_masks_with_coordinates(output_dir):
    masks_with_coords = []
    x_max_global, y_max_global = 0, 0
    for img_file in sorted(os.listdir(output_dir)):  # Ordenar os arquivos
        img_path = os.path.join(output_dir, img_file)
        
        # Extrair coordenadas diretamente do nome do arquivo
        try:
            x_min, y_min, x_max, y_max = img_file.replace(".png", "").split("_")
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        except ValueError:
            raise ValueError(f"Nome do arquivo '{img_file}' não segue o padrão esperado 'xmin_ymin_xmax_ymax.png'")
        
        # Atualizar os limites globais
        x_max_global = max(x_max_global, x_max)
        y_max_global = max(y_max_global, y_max)
        
        # Carregar a máscara
        mask = np.array(Image.open(img_path), dtype=np.uint8)
        masks_with_coords.append((mask, x_min, y_min, x_max, y_max))
    
    return masks_with_coords, (y_max_global, x_max_global)

# Carregar máscaras e calcular o tamanho da imagem final
masks_with_coords, dynamic_image_size = load_masks_with_coordinates(output_dir)

# Função para mesclar os tiles com base nas coordenadas
def merge_tiles(masks_with_coords, dynamic_image_size):
    # Criar uma matriz para a imagem mesclada
    merged_image = np.zeros(dynamic_image_size, dtype=np.float32)
    
    # Adicionar cada tile à posição correta na imagem final
    for mask, x_min, y_min, x_max, y_max in masks_with_coords:
        # Dimensões do tile
        tile_height, tile_width = mask.shape

        # Garantir compatibilidade entre dimensões esperadas e do tile
        valid_height = min(y_max - y_min, tile_height)
        valid_width = min(x_max - x_min, tile_width)

        # Inserir o tile na posição correta
        merged_image[y_min:y_min + valid_height, x_min:x_min + valid_width] += mask[:valid_height, :valid_width]
    
    # Normalizar sobreposições
    merged_image = np.clip(merged_image, 0, 1)  # Garantir valores no intervalo [0, 1]
    return merged_image

# Mesclar as máscaras
merged_image = merge_tiles(masks_with_coords, dynamic_image_size)

# Salvar a imagem final
merged_image_path = r"C:\Users\paulo\OneDrive\Documentos\SIACE\output_remontado_44_resnet.png"
Image.fromarray((merged_image * 255).astype(np.uint8)).save(merged_image_path)

print(f"Imagem mesclada salva em: {merged_image_path}")
