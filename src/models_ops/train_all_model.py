import os
import joblib  # Pour sauvegarder et recharger facilement les modèles
import numpy as np
import cv2  # Pour lire les images

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# === Fonctions ===

def charger_images_labels(paths_file, labels_file):
    with open(paths_file, 'r') as f:
        paths = f.read().splitlines()  # lit tous les chemins d’images
    with open(labels_file, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]  # récupère les labels associés

    images = []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # charge l’image en niveaux de gris
        if img is None:
            print(f"❌ Erreur image : {path}")
            continue
        images.append(img.flatten())  # transforme l’image 28x28 en vecteur 1D de 784 valeurs

    return np.array(images), np.array(labels)


# === Chargement des données MNIST ===

X_train, y_train = charger_images_labels(
    "resources/paths_mnist.txt",
    "resources/mnist_label.txt"
)

# === Définition des modèles ===

models = {
    "svm": SVC(probability=True),
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(),
    "knn": KNeighborsClassifier()
}

# === Entraînement et sauvegarde ===

output_dir = "models"
os.makedirs(output_dir, exist_ok=True)  # crée le dossier s'il n'existe pas

for name, model in models.items():
    print(f"\n🚀 Entraînement du modèle : {name}")
    model.fit(X_train, y_train)  # entraînement du modèle sur les données MNIST
    path = os.path.join(output_dir, f"{name}_model.pkl")
    joblib.dump(model, path)  # sauvegarde du modèle entraîné dans un fichier pickle
    print(f"✅ Modèle {name} sauvegardé dans {path}")