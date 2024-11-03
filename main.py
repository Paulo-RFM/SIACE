from U_net import unet_model, dice_loss
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping  # Importar EarlyStopping

# Diretórios dos tiles de imagem e máscara
image_tiles_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\tiles\images_tiles"
mask_tiles_dir = r"C:\Users\paulo\OneDrive\Documentos\SIACE\tiles\masks_tile"

# Parâmetros do modelo e dos dados
img_size = (256, 256)
batch_size = 8
epochs = 50

# Função para carregar e pré-processar uma única imagem e máscara
def load_image_and_mask(image_path, mask_path, img_size):
    img = Image.open(image_path).resize(img_size).convert("RGB")
    mask = Image.open(mask_path).resize(img_size).convert("L")

    img = np.array(img, dtype=np.float32) / 255.0
    mask = np.array(mask, dtype=np.float32) / 255.0
    mask = np.expand_dims(mask, axis=-1)  # Adicionar dimensão de canal para a máscara

    return img, mask

# Classe de Gerador Personalizado
class DataGenerator(Sequence):
    def __init__(self, image_files, mask_files, batch_size, img_size, shuffle=True):
        self.image_files = image_files
        self.mask_files = mask_files
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Número total de batches por época
        return int(np.floor(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        # Gera um batch de dados
        batch_image_files = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        batch_mask_files = self.mask_files[index * self.batch_size:(index + 1) * self.batch_size]

        images = []
        masks = []
        for img_file, mask_file in zip(batch_image_files, batch_mask_files):
            img, mask = load_image_and_mask(img_file, mask_file, self.img_size)
            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)

    def on_epoch_end(self):
        # Embaralhar os dados no final de cada época, se necessário
        if self.shuffle:
            combined = list(zip(self.image_files, self.mask_files))
            np.random.shuffle(combined)
            self.image_files, self.mask_files = zip(*combined)

# Listar todos os arquivos de imagem e máscara
image_files = sorted([os.path.join(image_tiles_dir, f) for f in os.listdir(image_tiles_dir) if f.endswith('.png')])
mask_files = sorted([os.path.join(mask_tiles_dir, f) for f in os.listdir(mask_tiles_dir) if f.endswith('.png')])

# Dividir os arquivos em conjuntos de treino e validação
image_train, image_val, mask_train, mask_val = train_test_split(image_files, mask_files, test_size=0.3, random_state=42)

# Instanciar geradores de treinamento e validação
train_generator = DataGenerator(image_train, mask_train, batch_size, img_size)
val_generator = DataGenerator(image_val, mask_val, batch_size, img_size)

# Instanciar e compilar o modelo U-Net
model = unet_model(input_size=(img_size[0], img_size[1], 3))
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

# Definir Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Treinamento do modelo com Early Stopping
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[early_stopping]  # Adicionar o callback de Early Stopping
)

# Salvar o modelo treinado
model.save("unet_tree_segmentation_model.h5")
