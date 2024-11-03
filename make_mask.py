import json
import numpy as np
import cv2
import os
from glob import glob
from PIL import Image

# Caminhos para os diretórios de entrada e saída
json_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp\labels"
output_mask_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp\masks"

# Criar o diretório de saída, se ele não existir
os.makedirs(output_mask_dir, exist_ok=True)

# Função para gerar máscaras binárias a partir dos arquivos JSON
def generate_masks(json_dir, output_mask_dir):
    json_files = glob(os.path.join(json_dir, "*.json"))
    
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        # Extrair as dimensões da imagem original
        image_height = data['imageHeight']
        image_width = data['imageWidth']

        # Inicializar uma máscara binária com dimensões da imagem
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # Percorrer cada anotação de polígono
        for shape in data['shapes']:
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)  # Preencher o polígono com valor 255 (branco)

        # Salvar a máscara binária com o mesmo nome do JSON
        mask_filename = os.path.splitext(os.path.basename(json_file))[0] + "_mask.png"
        mask_path = os.path.join(output_mask_dir, mask_filename)
        
        # Salvar a máscara como uma imagem PNG
        Image.fromarray(mask).save(mask_path)

        print(f"Máscara salva em: {mask_path}")

# Executar a função
generate_masks(json_dir, output_mask_dir)
