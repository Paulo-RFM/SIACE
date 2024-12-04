import os

def renomear_arquivos(diretorio):
    for root, dirs, files in os.walk(diretorio):
        for file in files:
            if "predict_tile_" in file:  # Verificar se o padrão esperado está presente no nome
                # Remover o prefixo 'predict_tile_' e manter apenas as coordenadas e tamanho
                novo_nome = file.replace("predict_tile_", "")
                antigo_caminho = os.path.join(root, file)
                novo_caminho = os.path.join(root, novo_nome)
                os.rename(antigo_caminho, novo_caminho)
                print(f"Renomeado: {antigo_caminho} para {novo_caminho}")
            else:
                print(f"Arquivo ignorado: {file}, não segue o padrão 'predict_tile_'.")

# Exemplo de uso
diretorio = r"C:\Users\paulo\OneDrive\Documentos\SIACE\predicts\44_resnet"
renomear_arquivos(diretorio)
