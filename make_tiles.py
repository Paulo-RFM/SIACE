from PIL import Image
import os
from glob import glob

Image.MAX_IMAGE_PIXELS = None

def split_image_and_mask_into_tiles(image_path, mask_path, tile_size=256, output_dir="tiles"):
    # Criar diretórios de saída para os tiles de imagens e máscaras
    os.makedirs(os.path.join(output_dir, "images_tiles"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks_tile"), exist_ok=True)

    # Carregar imagem e máscara
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    image_width, image_height = image.size

    # Contador para salvar os blocos com nomes únicos
    tile_count = 0

    for i in range(0, image_width, tile_size):
        for j in range(0, image_height, tile_size):
            # Definir as bordas do tile
            box = (i, j, i + tile_size, j + tile_size)

            # Cortar o tile da imagem e da máscara
            image_tile = image.crop(box)
            mask_tile = mask.crop(box)

            # Salvar o tile da imagem e da máscara com o mesmo índice
            image_tile.save(os.path.join(output_dir, "images_tiles", f"image_tile_{tile_count}.png"))
            mask_tile.save(os.path.join(output_dir, "masks_tile", f"mask_tile_{tile_count}.png"))
            
            tile_count += 1

    print(f"Dividido {image_path} e {mask_path} em {tile_count} blocos de {tile_size}x{tile_size} pixels.")

# Diretórios de imagens e máscaras
image_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp\Imagens_Drone"
mask_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp\masks"

# Padrão de busca para imagens e máscaras
image_files = sorted(glob(os.path.join(image_dir, "*.tif")))  # Ajuste a extensão se necessário
mask_files = sorted(glob(os.path.join(mask_dir, "*.png")))

# Certifique-se de que há correspondência entre as imagens e as máscaras
for image_file, mask_file in zip(image_files, mask_files):
    split_image_and_mask_into_tiles(image_file, mask_file, tile_size=256, output_dir="tiles")
