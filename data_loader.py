import os
import pandas as pd
import kagglehub

print("Baixando o dataset do IMDb...")
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_file))

print("\n--- Visualização dos Dados ---")
print(df.head())
print("\n--- Contagem de Sentimentos ---")
print(df["sentiment"].value_counts())
