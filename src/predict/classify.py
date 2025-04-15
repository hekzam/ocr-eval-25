import cv2
import numpy as np
from joblib import load
import os

# === Chargement des modèles entraînés
model_svm = load("models/svm_digit_model.joblib")
model_knn = load("models/knn_digit_model.joblib")

def load_and_prepare_image(image_path):
    """Charge une image 28x28 et la met au bon format pour les modèles"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Image non trouvée : {image_path}")
    
    img = img.astype("float32") / 255.0
    X = img.reshape(1, -1)  # pour scikit-learn
    return X

def classify_with_all_models(image_path):
    """Renvoie les prédictions de chaque modèle pour l'image donnée"""
    X = load_and_prepare_image(image_path)
    results = {}

    pred_svm = model_svm.predict(X)[0]
    proba_svm = model_svm.predict_proba(X)[0]
    results['SVM'] = (pred_svm, proba_svm)

    pred_knn = model_knn.predict(X)[0]
    proba_knn = model_knn.predict_proba(X)[0]
    results['KNN'] = (pred_knn, proba_knn)

    return results
