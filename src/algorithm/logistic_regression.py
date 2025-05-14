import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#besoin de normaliser les données
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "resources/models/lr_model.pkl"

def train(x_train, y_train):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    clf = LogisticRegression(max_iter=1000) # on rajoute cette option pour plus d'iter
    clf.fit(x_train, y_train)
    joblib.dump((clf, scaler), MODEL_PATH) # ici on sauvegarde le modele + le scaler
    print("Model trained and saved.")

def predict(x_test):
    clf, scaler = joblib.load(MODEL_PATH) # ici on récupère le tuple
    x_test = scaler.transform(x_test) # on normalise
    prediction_test = clf.predict_proba(x_test) # calculer les probas pour chaque classe

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

    x_train,y_train=df_train.iloc[:,1:].values,df_train.iloc[:,0].values
    x_test,y_test=df_test.iloc[:,1:].values,df_test.iloc[:,0].values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #enlever le # pour réentrainer le programme
    #train(x_train, y_train)

    rslt_rf = predict(x_test)

    print("", rslt_rf[0][1], np.sum(rslt_rf[0][1]))
    print("\ntest precision: ", accuracy_score(y_test, rslt_rf[1]))
    print("test matrice de confusion: \n", confusion_matrix(y_test, rslt_rf[1]))