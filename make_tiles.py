from PIL import Image
import os
from glob import glob

Image.MAX_IMAGE_PIXELS = None
# Diretórios de entrada
image_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp\Imagens_Drone"
mask_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp\masks"

# Diretórios de saída
base_output_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\tiles"
image_output_dir = os.path.join(base_output_dir, "images_tiles_512")
mask_output_dir = os.path.join(base_output_dir, "masks_tile_512")


def split_image_and_mask_into_tiles(image_path, mask_path, tile_size=512, image_output_dir="SAHI_img", mask_output_dir="SAHI_mask"):
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    image_output_dir = os.path.join(base_output_dir, "SAHI_img", base_name)
    mask_output_dir = os.path.join(base_output_dir, "SAHI_mask", base_name)
    
    # Criar diretórios de saída para os tiles de imagens e máscaras
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

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
            image_tile.save(os.path.join(image_output_dir, f"image_tile_{tile_count}.png"))
            mask_tile.save(os.path.join(mask_output_dir, f"mask_tile_{tile_count}.png"))
            
            tile_count += 1

    print(f"Dividido {image_path} e {mask_path} em {tile_count} blocos de {tile_size}x{tile_size} pixels.")

# Diretórios de imagens e máscaras
image_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp\Imagens_Drone"
mask_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp\masks"

# Padrão de busca para imagens e máscaras
image_files = glob(os.path.join(image_dir, "*.tif"))
mask_files = glob(os.path.join(mask_dir, "*.png"))

# Criar um dicionário para associar as imagens e máscaras pelo nome base
image_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
mask_dict = {os.path.splitext(os.path.basename(f))[0].replace("_mask", ""): f for f in mask_files}

# Diretórios de saída para os tiles de imagens e máscaras
image_output_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\tiles\SAHI_img"
mask_output_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\tiles\SAHI_mask"

# Percorrer os pares correspondentes e dividir em tiles
for key in image_dict.keys():
    if key in mask_dict:  # Garantir que uma máscara correspondente exista
        image_file = image_dict[key]
        mask_file = mask_dict[key]
        split_image_and_mask_into_tiles(image_file, mask_file, tile_size=512, image_output_dir=image_output_dir, mask_output_dir=mask_output_dir)
    else:
        print(f"Máscara correspondente para {key} não encontrada.")
