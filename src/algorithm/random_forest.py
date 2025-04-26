import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#TODO: stocker la version entrainée pour ne pas réentrainer a chaque fois et accélérer le temps d'exécution

def train_and_predict(x_train, y_train, x_test, y_test):

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)

    #predict value of label using classifier
    prediction_test = clf.predict(x_test)
    return prediction_test

if __name__ == "__main__":

    test_file = pd.read_csv("resources/test_data.csv")
    train_file = pd.read_csv("resources/train_data.csv")

    x_train,y_train=train_file.iloc[:,1:].values,train_file.iloc[:,0].values
    x_test,y_test=test_file.iloc[:,1:].values,test_file.iloc[:,0].values

    prediction_test = train_and_predict(x_train, y_train, x_test, y_test)

    
    print("\ntest precision: ", accuracy_score(y_test, prediction_test))
    print("test matrice de confusion: \n", confusion_matrix(y_test, prediction_test))