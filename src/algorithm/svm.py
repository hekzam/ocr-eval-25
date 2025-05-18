import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


def train(x_train, y_train, output, subset, c):
    # réduction du jeu d'entraînement
    if subset>0:
        x_train = x_train[:subset]
        y_train = y_train[:subset]
 # Création du modèle SVM avec :
    # - un noyau RBF (Radial Basis Function), qui est souvent le plus performant pour la classification de chiffres manuscrits (selon l'état de l'art)
    # - un paramètre de régularisation C = 50 (favorise une faible erreur d'entraînement)
    # - gamma = "scale", ce qui adapte automatiquement le gamma en fonction de la variance des données 
    clf = SVC(probability=True, kernel='rbf', C=c, gamma="scale")

    clf.fit(x_train, y_train)

    joblib.dump(clf, output)
    print("Modèle SVM entraîné et compressé")

def predict(x_test, input):
        # Chargement du modèle entraîné
    clf= joblib.load(input)
    prediction_test = clf.predict_proba(x_test)
    # Récupération de la classe avec la plus haute probabilité pour chaque exemple
    valeurProbable = []
    for i in range(len(prediction_test)):
        predicted_label = prediction_test[i].argmax()
        valeurProbable.append(predicted_label)

    return prediction_test, valeurProbable

if __name__ == "__main__":
    test_file = "resources/data_utilisees/test_data.parquet"
    train_file = "resources/data_utilisees/train_data.parquet"

    df_test = pd.read_parquet(test_file)
    df_train = pd.read_parquet(train_file)
    
    x_train, y_train = df_train.iloc[:, 1:].values, df_train.iloc[:, 0].values
    x_test, y_test = df_test.iloc[:, 1:].values, df_test.iloc[:, 0].values

    train(x_train, y_train)

    """
    prediction_test = predict(x_test)

    print("Probabilités d'un exemple :", prediction_test[0][1])
    print("\nPrécision sur le test :", accuracy_score(y_test, prediction_test[1]))
    print("Matrice de confusion :\n", confusion_matrix(y_test, prediction_test[1]))
    """