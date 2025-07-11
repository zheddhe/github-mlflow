import mlflow
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Chargement des données
# Remplacer avec le chemin vers votre jeu de données
print("Chargement des données...")
data = pd.read_csv("data/fake_data.csv")  # type: ignore
X = data.drop(columns=["date", "demand"])
X = X.astype('float')

# 2. Définir le chemin vers le modèle MLflow
# Remplacer avec le chemin vers votre dossier "rf_apples" créé précédemment
model_path = '/home/ubuntu/MLflow_Course/mlruns/637792679469892621/9f1a36320fb34910841e16fa7cef294d/artifacts/rf_apples'  # Par exemple : '/home/ubuntu/MLflow/mlruns/EXPERIMENT_ID/RUN_ID/artifacts/rf_apples'

# 3. Charger le modèle
print("Chargement du modèle...")
model = mlflow.sklearn.load_model(model_path)

# 4. Faire des prédictions sur l'ensemble du jeu de données
print("Calcul des prédictions...")
predictions = model.predict(X)

# 5. Calculer et afficher la moyenne des prédictions
# Calculer la moyenne des prédictions
mean_prediction = predictions.mean()  # Utiliser la fonction appropriée de numpy ou pandas

print(f"\nRésultats :")
print(f"Nombre de prédictions : {len(predictions)}")
print(f"Moyenne des prédictions : {mean_prediction:.2f}")