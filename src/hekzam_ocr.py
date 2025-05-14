import argparse
import pandas as pd
import time
import json
import os
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)

import algorithm.knn as knn
import algorithm.logistic_regression as lr
import algorithm.random_forest as rf
import algorithm.svm as svm
from utils.sauvegarde_csv import sauvegarder_resultats_csv


def calculer_metriques(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)
    }

def calcul_algo(nom_algo, module, x_test):
    print(f"Prédiction avec {nom_algo}...")
    start = time.perf_counter()
    y_pred = module.predict(x_test)
    stop = time.perf_counter()
    print(f"Prédiction terminée en {stop - start:.3f}s")
    return [nom_algo, y_pred, stop - start]

def executer_algo(nom, module, x_train, y_train, x_test, do_train):
    if do_train:
        start = time.perf_counter()
        module.train(x_train, y_train)
        stop = time.perf_counter()
        print(f"Modèle '{nom}' entraîné en {stop - start:.3f}s")

        # Sauvegarde temps dans fichier JSON
        suffix = nom.lower().replace(" ", "_")
        with open(f"results/temp/temps_train_{suffix}.json", "w") as f:
            json.dump({"temps": stop - start}, f)

        return None  # pas de prédiction

    return calcul_algo(nom, module, x_test)

def main():
    parser = argparse.ArgumentParser(description='detect numbers from images')
    parser.add_argument('algorithm', metavar='algorithm', help='choose the algorithm to use',
                        choices=['knn', 'svm', 'rf', 'lr', 'everything'])
    parser.add_argument('--train', help='specify whether to train the model',
                        action='store_true')
    args = parser.parse_args()

    df_test = pd.read_parquet("resources/data_utilisees/test_data.parquet")
    df_train = pd.read_parquet("resources/data_utilisees/train_data.parquet")

    x_train, y_train = df_train.iloc[:, 1:].values, df_train.iloc[:, 0].values
    x_test, y_test = df_test.iloc[:, 1:].values, df_test.iloc[:, 0].values

    rslt = []

    match args.algorithm:
        case 'knn':
            r = executer_algo("knn", knn, x_train, y_train, x_test, args.train)
            if r: rslt.append(r)
        case 'svm':
            r = executer_algo("svm", svm, x_train, y_train, x_test, args.train)
            if r: rslt.append(r)
        case 'rf':
            r = executer_algo("random forest", rf, x_train, y_train, x_test, args.train)
            if r: rslt.append(r)
        case 'lr':
            r = executer_algo("logistic regression", lr, x_train, y_train, x_test, args.train)
            if r: rslt.append(r)
        case 'everything':
            for nom, module in [
                ("knn", knn),
                ("svm", svm),
                ("random forest", rf),
                ("logistic regression", lr)
            ]:
                r = executer_algo(nom, module, x_train, y_train, x_test, args.train)
                if r: rslt.append(r)

    # Si on est en mode test (pas juste entraînement)
    resultats_export = []

    for nom_algo, predictions, temps in rslt:
        y_pred = predictions[1] if isinstance(predictions, tuple) else predictions
        cm = confusion_matrix(y_test, y_pred)
        metriques = calculer_metriques(y_test, y_pred)

        print(f"\nRésultats pour {nom_algo} (temps de test : {temps:.3f}s)")
        for nom, val in metriques.items():
            print(f"  - {nom:<18}: {val:.4f}")

        resultats_export.append({
            "nom": nom_algo,
            "temps": temps,
            "metriques": metriques,
            "matrice_confusion": cm.tolist()
        })

        # Sauvegarde matrice de confusion
        suffix = nom_algo.lower().replace(" ", "_")
        filename = f"results/confusion_matrices/matrice_confusion_{suffix}.json"
        with open(filename, "w") as f:
            json.dump(cm.tolist(), f, indent=4)
        print(f"Matrice enregistrée : {filename}")

        # Lecture du temps d'entraînement depuis le fichier JSON
        try:
            with open(f"results/temp/temps_train_{suffix}.json") as f:
                temps_train = json.load(f)["temps"]
            os.remove(f"results/temp/temps_train_{suffix}.json")
        except FileNotFoundError:
            temps_train = 0.0

        # Sauvegarde CSV
        sauvegarder_resultats_csv(
            nom_algo=nom_algo,
            metriques=metriques,
            temps_train=temps_train,
            temps_test=temps,
            n_train=10000 if nom_algo == "svm" else len(x_train),
            n_test=len(x_test)
        )


if __name__ == "__main__":
    main()
