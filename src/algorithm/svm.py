import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


MODEL_PATH = "resources/models/svm_model.pkl"

def train(x_train, y_train):
    clf = SVC(probability=True)
    clf.fit(x_train, y_train)
    joblib.dump(clf, MODEL_PATH)
    print("Model trained and saved.")

def predict(x_test):
    clf = joblib.load(MODEL_PATH)
    prediction_test = clf.predict_proba(x_test)

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


    x_train, y_train = train_file.iloc[:, 1:].values, train_file.iloc[:, 0].values
    x_test, y_test = test_file.iloc[:, 1:].values, test_file.iloc[:, 0].values

    train(x_train, y_train)
    prediction_test = predict(x_test)

    print("", prediction_test[0][1])
    print("\nprécision sur les données de test :", accuracy_score(y_test, prediction_test[1]))
    print("matrice de confusion :\n", confusion_matrix(y_test, prediction_test[1]))