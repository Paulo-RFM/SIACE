import json
import numpy as np
import cv2
import os
from PIL import Image
from glob import glob

# Caminhos para os diretórios de entrada e saída
json_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp\labels"
output_mask_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\imagens_rp\masks_colored"

# Criar o diretório de saída, se ele não existir
os.makedirs(output_mask_dir, exist_ok=True)

# Definindo cores para cada espécie (group_id)
# Exemplo: [R, G, B]
colors = {
    1: [255, 0, 0],       # Vermelho para a espécie 1
    2: [0, 255, 0],       # Verde para a espécie 2
    3: [0, 0, 255],       # Azul para a espécie 3
    4: [255, 255, 0],     # Amarelo para a espécie 4
    None: [128, 128, 128]  # Cinza para árvores não identificadas (group_id = null)
}

# Função para gerar máscaras coloridas usando group_id e shape_type
def generate_colored_masks(json_dir, output_mask_dir):
    json_files = glob(os.path.join(json_dir, "*.json"))
    
    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        # Extrair as dimensões da imagem original
        image_height = data['imageHeight']
        image_width = data['imageWidth']

        # Inicializar uma máscara colorida (3 canais para RGB)
        mask = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        # Percorrer cada anotação e desenhar com base no shape_type
        for shape in data['shapes']:
            # Obter o group_id para determinar a espécie, se presente
            group_id = shape.get("group_id", None)
            
            # Usar a cor correspondente à espécie ou cor para não identificado
            color = colors.get(group_id, colors[None])
            
            # Verificar o tipo de forma (rectangle ou polygon)
            shape_type = shape.get("shape_type", "polygon")
            points = np.array(shape['points'], dtype=np.int32)

            if shape_type == "rectangle":
                # Se for um retângulo, definir as coordenadas do canto superior esquerdo e inferior direito
                (x1, y1), (x2, y2) = points
                cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=-1)
            elif shape_type == "polygon":
                # Se for um polígono, preencher com a cor específica
                cv2.fillPoly(mask, [points], color)

        # Salvar a máscara colorida com um nome correspondente ao JSON
        mask_filename = os.path.splitext(os.path.basename(json_file))[0] + "_mask_colored.png"
        mask_path = os.path.join(output_mask_dir, mask_filename)
        
        # Salvar a máscara como uma imagem PNG
        Image.fromarray(mask).save(mask_path)

        print(f"Máscara salva em: {mask_path}")

# Executar a função
generate_colored_masks(json_dir, output_mask_dir)
