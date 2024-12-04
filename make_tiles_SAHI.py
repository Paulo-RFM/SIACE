import os
from sahi.slicing import slice_image

# Diretórios das imagens e máscaras
image_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp_crop"
mask_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\mask_rp_crop"

# Diretórios de saída para os tiles
output_image_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\tiles\images_tiles_512_crop"
output_mask_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\tiles\masks_tile_512_crop"

# Garantir que os diretórios de saída existem
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Função para processar as imagens e máscaras
def process_images_and_masks(image_dir, mask_dir, output_image_dir, output_mask_dir):
    # Listar arquivos de imagem e máscara
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith((".png", ".jpg", ".jpeg"))])

    if len(image_files) != len(mask_files):
        print("Aviso: O número de imagens não corresponde ao número de máscaras!")

    for image_file, mask_file in zip(image_files, mask_files):
        # Caminhos completos
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)

        # Verificar se as dimensões da imagem e da máscara são idênticas
        if not verify_dimensions(image_path, mask_path):
            print(f"As dimensões de {image_file} e {mask_file} não correspondem. Pulando...")
            continue

        # Diretórios de saída específicos para a imagem e máscara
        img_output_dir = os.path.join(output_image_dir, os.path.splitext(image_file)[0])
        mask_output_dir = os.path.join(output_mask_dir, os.path.splitext(mask_file)[0])

        # Garantir que os diretórios de saída existem
        os.makedirs(img_output_dir, exist_ok=True)
        os.makedirs(mask_output_dir, exist_ok=True)

        # Processar imagem
        print(f"Processando imagem: {image_file}")
        img_slice_result = slice_image(
            image=image_path,
            output_file_name="tile",
            output_dir=img_output_dir,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=True
        )

        # Processar máscara
        print(f"Processando máscara: {mask_file}")
        mask_slice_result = slice_image(
            image=mask_path,
            output_file_name="tile",
            output_dir=mask_output_dir,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=True
        )

        # Verificar correspondência de tiles
        img_tiles = len(img_slice_result.sliced_image_list)
        mask_tiles = len(mask_slice_result.sliced_image_list)

        if img_tiles != mask_tiles:
            print(f"Atenção: {img_tiles} tiles gerados para {image_file}, mas {mask_tiles} tiles gerados para {mask_file}.")
        else:
            print(f"Tiles correspondentes gerados para {image_file} e {mask_file}.")

# Função para verificar se as dimensões de imagem e máscara são iguais
def verify_dimensions(image_path, mask_path):
    from PIL import Image

    with Image.open(image_path) as img, Image.open(mask_path) as mask:
        return img.size == mask.size

# Executar o processamento
process_images_and_masks(image_dir, mask_dir, output_image_dir, output_mask_dir)
