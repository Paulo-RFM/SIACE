import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

# Recriar o modelo e carregar os pesos
# Definir o dispositivo (CPU ou GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Diretório contendo os tiles
img_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\tiles\images_tiles_512_crop\44"

# Transformação para converter a imagem em tensor
transform = transforms.Compose([
    transforms.ToTensor(),  # Converte imagem para tensor (C, H, W) com valores entre 0 e 1
])

# Lista para armazenar resultados
results = []

# Iterar sobre as imagens no diretório
for img_file in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_file)  # Caminho completo para o arquivo de imagem

    # Abrir a imagem usando PIL
    tile = Image.open(img_path).convert("RGB")  # Converter para RGB se necessário

    # Transformar a imagem em tensor
    tile_tensor = transform(tile)
    tile_tensor = tile_tensor.unsqueeze(0)  # Adicionar dimensão do batch (1, C, H, W)

    # Fazer inferência com o modelo
    with torch.no_grad():
        prediction = model(tile_tensor)

    # Salvar o resultado da predição
    results.append(prediction.squeeze(0))  # Remover dimensão do batch (C, H, W)

# Exibir o número total de predições realizadas
print(f"Total de predições realizadas: {len(results)}")
