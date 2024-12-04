import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# Diretórios de entrada e saída
img_input_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp\Imagens_Drone"
mask_input_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp\masks"
img_output_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp_crop"
mask_output_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\mask_rp_crop"


def crop_image_by_proportion(image_path, proportion, output_dir):
    # Carregar a imagem
    image = Image.open(image_path)
    width, height = image.size

    # Calcular a quantidade de pixels a serem cortados com base na proporção
    border_width = int(width * proportion)
    border_height = int(height * proportion)

    # Calcular as coordenadas da região central
    left = border_width
    top = border_height
    right = width - border_width
    bottom = height - border_height

    # Recortar a região central
    cropped_image = image.crop((left, top, right, bottom))

    # Salvar a imagem recortada no subdiretório apropriado
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    #subdir_path = os.path.join(output_dir, str(subdir_index))
    #os.makedirs(subdir_path, exist_ok=True)
    output_path = os.path.join(output_dir, base_name + '.png')
    cropped_image.save(output_path)
    print(f"Salvou: {output_path}")

def process_images_and_masks(img_input_dir, mask_input_dir, proportion, img_output_dir, mask_output_dir):
    # Garantir que os diretórios de saída existam
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    # Inicializar o índice dos subdiretórios
    subdir_index = 1

    # Percorrer todos os arquivos na pasta de entrada das imagens
    for img_filename in os.listdir(mask_input_dir):
        # Verificar se o arquivo é uma imagem (pode ser adaptado conforme necessário)
        if img_filename.lower().endswith(('.png', '.tif')):
            img_path = os.path.join(img_input_dir, img_filename)
            
            # Supondo que as máscaras têm o mesmo nome base das imagens
            base_name = os.path.splitext(img_filename)[0]
            mask_filename = base_name + ".png"
            mask_path = os.path.join(mask_input_dir, mask_filename)
            
            if os.path.exists(mask_path):  # Verifica se a máscara correspondente existe
               # crop_image_by_proportion(img_path, proportion, img_output_dir)
                crop_image_by_proportion(mask_path, proportion, mask_output_dir)

# Proporção da borda a ser cortada (por exemplo, 10% das dimensões da imagem)
proportion = 0.1

# Processar todas as imagens e suas máscaras correspondentes na pasta de entrada
process_images_and_masks(img_input_dir, mask_input_dir, proportion, img_output_dir, mask_output_dir)
