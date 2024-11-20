import os
from glob import glob
from sahi.slicing import slice_image

# Diretórios de entrada
image_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp\imagens_rp_crop"
mask_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp\mask_rp_crop"

# Diretórios de saída
base_output_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\tiles"
image_output_dir = os.path.join(base_output_dir, "images_tiles_512_crop")
mask_output_dir = os.path.join(base_output_dir, "masks_tile_512_crop")

# Criar diretórios de saída, se necessário
os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)

# Padrão de busca para imagens e máscaras
image_files = glob(os.path.join(image_dir, "*.png"))
mask_files = glob(os.path.join(mask_dir, "*.png"))

# Dicionários para associar imagens e máscaras pelo nome base
image_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
mask_dict = {os.path.splitext(os.path.basename(f))[0].replace("_mask", ""): f for f in mask_files}

# Função para cortar imagens e máscaras com SAHI
def slice_image_and_mask(image_path, mask_path, tile_size=512, overlap_ratio=0.2):
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Diretórios específicos para cada imagem/máscara
    image_out_dir = os.path.join(image_output_dir, base_name)
    mask_out_dir = os.path.join(mask_output_dir, base_name)

    # Criar diretórios de saída
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    # Cortar imagem
    slice_image(
        image=image_path,
        output_file_name=image_files,
        output_dir=image_out_dir,
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
    )

    # Cortar máscara
    slice_image(
        image=mask_path,
        output_file_name=mask_file, 
        output_dir=mask_out_dir,
        slice_height=tile_size,
        slice_width=tile_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
    )

    print(f"Cortadas {image_path} e {mask_path} em tiles de {tile_size}x{tile_size} pixels com sobreposição de {overlap_ratio * 100}%.")

# Percorrer os pares correspondentes e dividir em tiles
for key in image_dict.keys():
    if key in mask_dict:  # Garantir que uma máscara correspondente exista
        image_file = image_dict[key]
        mask_file = mask_dict[key]
        slice_image_and_mask(image_file, mask_file, tile_size=512, overlap_ratio=0.2)
    else:
        print(f"Máscara correspondente para {key} não encontrada.")
