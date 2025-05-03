import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#besoin de normaliser les données 
from sklearn.preprocessing import StandardScaler

#TODO: stocker la version entrainée pour ne pas réentrainer a chaque fois et accélérer le temps d'exécution

MODEL_PATH = "resources/models/lr_model.pkl"

def train(x_train, y_train):
    #max_iter créé une erreur comme étant trop bas, mettre en param "max_iter=10000" 
    #pour ne plus avoir l'erreur, semble avoir un impact negatif sur le code
    clf = LogisticRegression()
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

    test_file = pd.read_csv("resources/test_data.csv")
    train_file = pd.read_csv("resources/train_data.csv")

    x_train,y_train=train_file.iloc[:,1:].values,train_file.iloc[:,0].values
    x_test,y_test=test_file.iloc[:,1:].values,test_file.iloc[:,0].values

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    #enlever le # pour réentrainer le programme 
    train(x_train, y_train)

    rslt_rf = predict(x_test)

    print("", rslt_rf[0][1])
    print("\ntest precision: ", accuracy_score(y_test, rslt_rf[1]))
    print("test matrice de confusion: \n", confusion_matrix(y_test, rslt_rf[1]))