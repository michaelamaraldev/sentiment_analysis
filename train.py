import os
import joblib
import pandas as pd
import kagglehub
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    return text.lower()

path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_file))

print("Limpando os textos...")
df["review_clean"] = df["review"].apply(clean_text)

X = df["review_clean"]
y = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Treinando o modelo...")
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print(f"\nAcurácia: {accuracy_score(y_test, y_pred):.2f}")
print("\nRelatório:\n", classification_report(y_test, y_pred))

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/logistic_model.joblib')
joblib.dump(vectorizer, 'models/vectorizer.joblib')
print("Modelo e Vetorizador salvos na pasta 'models/'!")