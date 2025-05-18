import os
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# valeurs choisies à partir de knn_search
K_VALUE = 3

def train(x_train, y_train, output, subset):
    # réduction du jeu d'entraînement
    if subset>0:
        x_train = x_train[:subset]
        y_train = y_train[:subset]

    os.makedirs(os.path.dirname(output), exist_ok=True)
    joblib.dump((x_train, y_train, K_VALUE), output, compress=3)
    print(f"Données KNN sauvegardées (k={K_VALUE}, n={len(x_train)})")

def predict(x_test, input):
    x_train, y_train, _ = joblib.load(input)
    print(x_train.shape[1], x_test.shape[1] ) 
    knn_classifier = KNeighborsClassifier(n_neighbors=K_VALUE)
    knn_classifier.fit(x_train, y_train)
    prediction_test = knn_classifier.predict(x_test)
    return prediction_test

if __name__ == "__main__":

    test_file = "resources/data_utilisees/test_data.parquet"
    train_file = "resources/data_utilisees/train_data.parquet"

    df_test = pd.read_parquet(test_file)
    df_train = pd.read_parquet(train_file)

    x_train, y_train = df_train.iloc[:, 1:].values, df_train.iloc[:, 0].values
    x_test, y_test = df_test.iloc[:, 1:].values, df_test.iloc[:, 0].values

    train(x_train, y_train)

    y_pred = predict(x_test)

    print("\nprécision :", accuracy_score(y_test, y_pred))
    print("matrice de confusion :\n", confusion_matrix(y_test, y_pred))
