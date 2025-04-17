import os
import joblib
import numpy as np
import cv2
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# === Utilitaires ===

def charger_images_labels(paths_file, labels_file):
    with open(paths_file, 'r') as f:
        paths = f.read().splitlines()
    with open(labels_file, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]

    images = []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Image manquante ou illisible : {path}")
            continue
        images.append(img.flatten())  # 28x28 => 784

    return np.array(images), np.array(labels)


# === Chargement des données CUSTOM ===

X_custom, y_custom = charger_images_labels(
    "resources/paths_custom.txt",
    "resources/custom_label.txt"
)

# === Liste des modèles ===

model_names = ["svm", "logreg", "rf", "knn"]
model_dir = "models"

# === Évaluation ===

print("\n📊 Évaluation des modèles sur les données CUSTOM :")

for name in model_names:
    model_path = os.path.join(model_dir, f"{name}_model.pkl")
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable : {model_path}")
        continue

    print(f"\n🔹 {name.upper()} :")
    model = joblib.load(model_path)
    y_pred = model.predict(X_custom)

    accuracy = accuracy_score(y_custom, y_pred)
    print(f"🎯 Accuracy : {accuracy:.4f}")
    print(classification_report(y_custom, y_pred, digits=3))
    
    # afficher la matrice de confusion :
    # print("🧮 Matrice de confusion :")
    # print(confusion_matrix(y_custom, y_pred))
