import kagglehub
import pandas as pd
import os

print("Baixando o dataset do IMDb...")
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

arquivos = os.listdir(path)
csv_nome = [f for f in arquivos if f.endswith(".csv")][0]
caminho_completo = os.path.join(path, csv_nome)

df = pd.read_csv(caminho_completo)

print("\n--- Visualização dos Dados ---")
print(df.head())

print("\n--- Contagem de Sentimentos ---")
print(df["sentiment"].value_counts())
