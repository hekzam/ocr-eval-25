import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# TODO: stocker la version entraînée

def train_and_predict(x_train, y_train, x_test, y_test):
    # créer le modele
    model = SVC(probability=True)
    model.fit(x_train, y_train)

    prediction_test = model.predict(x_test)
    return prediction_test

if __name__ == "__main__":

    test_file = pd.read_csv("resources/test_data.csv")
    train_file = pd.read_csv("resources/train_data.csv")

    x_train, y_train = train_file.iloc[:, 1:].values, train_file.iloc[:, 0].values
    x_test, y_test = test_file.iloc[:, 1:].values, test_file.iloc[:, 0].values

    prediction_test = train_and_predict(x_train, y_train, x_test, y_test)

    print("\nprécision sur les données de test :", accuracy_score(y_test, prediction_test))
    print("matrice de confusion :\n", confusion_matrix(y_test, prediction_test))