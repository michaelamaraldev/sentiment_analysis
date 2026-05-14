import kagglehub
import pandas as pd
import os

print("Baixando dataset...")
path = kagglehub.dataset_download("sahilislam007/letterbox-movie-classification-dataset")

arquivos = os.listdir(path)
csv_nome = [f for f in arquivos if f.endswith('.csv')][0]
caminho_completo = os.path.join(path, csv_nome)

df = pd.read_csv(caminho_completo)
print("\n--- Colunas encontradas no dataset ---")
print(df.columns.tolist())
print("\n--- Exemplo dos dados (5 primeiras linhas) ---")
print(df.head())