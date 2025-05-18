import pandas as pd
import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def train(x_train, y_train, output, subset):
    # réduction du jeu d'entraînement
    if subset>0:
        x_train = x_train[:subset]
        y_train = y_train[:subset]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    clf = LinearSVC()
    clf.fit(x_train, y_train)

    joblib.dump((clf, scaler), output)
    print("Modèle LinearSVC entraîné et sauvegardé.")

def predict(x_test, input):
    clf, scaler = joblib.load(input)
    x_test = scaler.transform(x_test)

    valeurProbable = clf.predict(x_test)
    return valeurProbable

if __name__ == "__main__":
    test_file = "resources/data_utilisees/test_data.parquet"
    train_file = "resources/data_utilisees/train_data.parquet"

    df_test = pd.read_parquet(test_file)
    df_train = pd.read_parquet(train_file)

    x_train, y_train = df_train.iloc[:, 1:].values, df_train.iloc[:, 0].values
    x_test, y_test = df_test.iloc[:, 1:].values, df_test.iloc[:, 0].values

    train(x_train, y_train)
    y_pred = predict(x_test)

    print("Précision sur les données de test :", accuracy_score(y_test, y_pred))
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))
