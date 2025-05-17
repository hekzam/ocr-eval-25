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

def total_data():
    mnist_label = "resources/mnist_label.txt"
    custom_label = "resources/custom_label.txt" 
    with open(mnist_label) as f:
        count = sum(1 for _ in f)
    with open(custom_label) as f:
        count += sum(1 for _ in f)
    return count

def authorized_ratio(entry):
    try:
        entry = float(entry)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{entry} is not a float")
    if entry < 0.1 or entry > 0.99:
        raise argparse.ArgumentTypeError(f"{entry} is not in range 0.1-0.99")
    return entry

def authorized_size(entry):
    try:
        entry = int(entry)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{entry} is not an integer")
    if entry < 2 or entry > total_data():
        raise argparse.ArgumentTypeError(f"{entry} is not in range 2-"+total_data())
    return entry

def build_model_output(entry):
    return entry+"/trained_model.pkl"

def build_train_output(entry):
    return entry+"/train_data.parquet"

def build_test_output(entry):
    return entry+"/test_data.parquet"

def build_result_output(entry):             #TODO: mettre un bon nom de variable de sortie 
    return entry+"/trained_model.pkl"

def validate(entry, list_valid):
    valid_option = list_valid
    rslt = entry.split(',')
    for f in rslt:
        if f not in valid_option:
            raise argparse.ArgumentTypeError(f"Invalid feature: {f}")
    return '+'.join(rslt)

def valid_features(entry):
    return validate(entry, {"flatten", "zoning", "4lrp"})

def valid_models(entry):
    return validate(entry, {"knn", "svm", "rf", "lr"})


def main():
    
    #permet de vérifier si training est activé pour faire des options seulement si il l'est
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('train', help= 'specify whether to train the model',
                        type=valid_models, default=None)
    pre_parser.add_argument('model', help= 'choose the algorithm to use',
                        type=valid_models, default = None)
    pre_parser.add_argument('rebuild_data', help = 'restart building the data, allow changes in features and dataset used in next execution',
                        action='store_true')
    pre_args,_ = pre_parser.parse_known_args()

    #parser complet
    parser = argparse.ArgumentParser(description='detect numbers from images', 
                                     parents=[pre_parser])
    
    #ajoute les arguments de rebuild data
    if pre_args.rebuild_data:
        parser.add_argument('--custom_train_output',
                            help='choose where the train data will be stored',
                            type=build_train_output,
                            default="resources/data_utilisees/train_data.parquet")
        parser.add_argument('--custom_test_output',
                            help='choose where the test data will be stored',
                            type=build_test_output,
                            default="resources/data_utilisees/test_data.parquet")
        parser.add_argument('--features', help="choose the method used to extract features",
                            type=valid_features, default="flatten+zoning")
        parser.add_argument('--training_ratio', 
                            help='choose how many of the data will be used to train, the remaining will be used in test. from 0.1 to 0.99',
                            type=authorized_ratio, default = 0.8)
        parser.add_argument('--data_size', 
                            help='choose the size of the used data',
                            type= authorized_size, default= total_data())
        
    #ajoute les arguments de train
    elif pre_args.train!=None:
        parser.add_argument('--custom_input')
        parser.add_argument('--custom_output', help='choose where the model is stored, the file will be created',
                            type= build_result_output,
                            default="results/confusion_matrices/")
        match pre_args.train:           #TODO: ajouter les implementations custom
            case 'knn':
                print()
            case 'svm':
                print()
            case 'lr':
                print()
            case 'rf':
                print()

    #ajoute les arguments de l'execution des models
    elif pre_args.model!=None:
        parser.add_argument('--custom_input', help='choose where the model is stored',
                            default="resources/models/"+ pre_args.model +"_model.pkl")
        parser.add_argument('--custom_output', help='choose where the model is stored',
                            type= build_model_output,
                            default="resources/models/"+ pre_args.model +"_model.pkl")

    args = parser.parse_args()
    print(args)


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
        filename = args.custom_output+"matrice_confusion_{suffix}.json"
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
