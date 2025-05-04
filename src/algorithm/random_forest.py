import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#TODO: mesurer le temps d'execution 

MODEL_PATH = "resources/models/rf_model.pkl"

def train(x_train, y_train):
    clf = RandomForestClassifier()
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

    test_file = "resources/data_utilisees/test_data.pkl"
    train_file = "resources/data_utilisees/train_data.pkl"

    with open(test_file, "rb") as f:
        df_test = pickle.load(f)
    with open(train_file, "rb") as f:
        df_train = pickle.load(f)
    
    x_train,y_train=df_train.iloc[:,1:].values,df_train.iloc[:,0].values
    x_test,y_test=df_test.iloc[:,1:].values,df_test.iloc[:,0].values

    #enlever le # pour réentrainer le programme 
    #train(x_train, y_train)

    rslt_rf = predict(x_test)

    print("", rslt_rf[0][1])
    print("\ntest precision: ", accuracy_score(y_test, rslt_rf[1]))
    print("test matrice de confusion: \n", confusion_matrix(y_test, rslt_rf[1]))