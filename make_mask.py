import json
import numpy as np
import cv2
import os
from glob import glob
from PIL import Image

# Caminhos para os diretórios de entrada e saída
json_dir = r"/home/paulomaciel/proj/labels2/"
output_mask_dir = r"/home/paulomaciel/proj/masks"


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

        # Percorrer cada anotação e desenhar a forma correspondente
        for shape in data['shapes']:
            shape_type = shape.get("shape_type", "polygon")  # Tipo da forma (padrão: polígono)
            points = np.array(shape['points'], dtype=np.int32)

            if shape_type == "polygon":
                cv2.fillPoly(mask, [points], 255)  # Preencher o polígono com 255 (branco)
            elif shape_type == "rectangle":
                # Pega os dois pontos do retângulo
                top_left = tuple(points[0])
                bottom_right = tuple(points[1])
                cv2.rectangle(mask, top_left, bottom_right, 255, -1)  # Retângulo preenchido
            elif shape_type == "circle":
                # Círculo usa o primeiro ponto como centro e o segundo ponto para o raio
                center = tuple(points[0])
                radius = int(np.linalg.norm(points[1] - points[0]))  # Distância entre dois pontos
                cv2.circle(mask, center, radius, 255, -1)  # Círculo preenchido

        # Salvar a máscara binária com o mesmo nome do JSON
        mask_filename = os.path.splitext(os.path.basename(json_file))[0] + ".png"
        mask_path = os.path.join(output_mask_dir, mask_filename)
        
        # Salvar a máscara como uma imagem PNG
        Image.fromarray(mask).save(mask_path)

        print(f"Máscara salva em: {mask_path}")

# Executar a função
generate_masks(json_dir, output_mask_dir)
