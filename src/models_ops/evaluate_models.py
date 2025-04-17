import os
import joblib  # Pour charger les modèles entraînés au format .pkl
import numpy as np
import cv2  # Pour lire les images PNG
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Pour l’évaluation

# === Fonction utilitaire pour charger les images et labels ===
def charger_images_labels(paths_file, labels_file):
    # Lecture des chemins des images
    with open(paths_file, 'r') as f:
        paths = f.read().splitlines()
    # Lecture des labels correspondants
    with open(labels_file, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    
    images = []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Lecture image en niveaux de gris
        if img is None:
            print(f"❌ Erreur image : {path}")
            continue
        images.append(img.flatten())  # Image 28x28 transformée en vecteur de 784 pixels

    return np.array(images), np.array(labels)

# Chargement des données MNIST
X, y = charger_images_labels("resources/paths_mnist.txt", "resources/mnist_label.txt")

# Noms des modèles à charger
model_names = ["svm", "logreg", "rf", "knn"]

# === Évaluation de chaque modèle ===
print("\\n📊 Évaluation des modèles sur les données MNIST :")
for name in model_names:
    path = f"models/{name}_model.pkl"
    if not os.path.exists(path):
        print(f"❌ Modèle manquant : {name}")
        continue
    model = joblib.load(path)  # Chargement du modèle
    y_pred = model.predict(X)  # Prédiction sur les images MNIST
    acc = accuracy_score(y, y_pred)  # Calcul de la précision
    print(f"\\n🔹 {name.upper()} :")
    print(f"   🎯 Accuracy : {acc:.4f}")
    print(classification_report(y, y_pred, digits=3))  # Rapport complet par classe
    # print(confusion_matrix(y, y_pred))  # (Optionnel) Matrice de confusion