import argparse
import pandas as pd
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

import utils.main_tools as tools
from utils.sauvegarde_csv import sauvegarder_resultats_csv


def calculer_metriques(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)
    }


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
        raise argparse.ArgumentTypeError(f"{entry} is not in range 2-{total_data()}")
    return entry


def is_directory(entry):
    if not os.path.isdir(entry):
        raise argparse.ArgumentTypeError(f"'{entry}' is not a valid directory.")
    return entry+"/"

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
    return validate(entry, {"knn", "svm", "rf", "lr", "linear_svm"})



def main():
    
    #permet de vérifier si training est activé pour faire des options seulement si il l'est
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--train', help= 'choose to train a model',
                        type=valid_models, default=None)
    pre_parser.add_argument('--model', help= 'choose the model to use',
                        type=valid_models, default = None)
    pre_parser.add_argument('--rebuild_data', help = 'restart building the data, allow changes in features and dataset used in next execution',
                        action='store_true')
    pre_args,_ = pre_parser.parse_known_args()

    #parser complet
    parser = argparse.ArgumentParser(description='detect numbers from images', 
                                     parents=[pre_parser])
    
    #ajoute les arguments de rebuild data
    if pre_args.rebuild_data:
        parser.add_argument('--r_features', help="choose the method used to extract features",
                            type=valid_features, default="flatten,zoning")
        parser.add_argument('--r_training_ratio', 
                            help='choose how many of the data will be used to train, the remaining will be used in test. from 0.1 to 0.99',
                            type=authorized_ratio, default = 0.8)
        parser.add_argument('--r_data_size', 
                            help='choose the size of the used data',
                            type= authorized_size, default= total_data())
        
    #ajoute les arguments de train
    if pre_args.train!=None:
        parser.add_argument('--t_custom_output', help='choose where the model is stored, the file will be created',
                            type= is_directory,
                            default="results/models")
        if 'knn' in pre_args.train:
            parser.add_argument('--knn_subset', help='choose the number of data used to train the model, -1 to use everything', 
                                type = int, default = 60000)
        if 'svm' in pre_args.train:
            parser.add_argument('--svm_subset', help='choose the number of data used to train the model, -1 to use everything', 
                                type = int, default = 10000)
            parser.add_argument('--svm_regulation_parameter', help='choose the regulation parameter',
                                type = int, default=50)
        if 'linear_svm' in pre_args.train:
            parser.add_argument('--lsvm_subset', help='choose the number of data used to train the model, -1 to use everything', 
                                type = int, default = 10000)
        if 'rf' in pre_args.train:
            parser.add_argument('--rf_subset', help='choose the number of data used to train the model, -1 to use everything', 
                                type = int, default = -1)
        if 'lr' in pre_args.train:
            parser.add_argument('--lr_subset', help='choose the number of data used to train the model, -1 to use everything', 
                                type = int, default = -1)
            

    #ajoute les arguments de l'execution des models
    #TODO, faire en sorte que l'adresse ne comporte pas de fichier dans le resultat, possibilité de plusieurs model exec
    if pre_args.model!=None:
        parser.add_argument('--m_custom_input', help='choose where the model is picked-up, doesnt work if the model have a custom name',
                            type=is_directory,
                            default="results/confusion_matrices")
        parser.add_argument('--m_custom_output', help='choose where the result is stored',
                            type=is_directory,
                            default="results/results_algo")

    args = parser.parse_args()
    print(args)

    """
            Execution du code
    """

    if pre_args.rebuild_data:
        tools.construction_donnees(args)
    
    
    df_train = pd.read_parquet("resources/data_utilisees/train_data.parquet")    
    df_test = pd.read_parquet("resources/data_utilisees/test_data.parquet")
    x_test, y_test = df_test.iloc[:, 1:].values, df_test.iloc[:, 0].values
    x_train, y_train = df_train.iloc[:, 1:].values, df_train.iloc[:, 0].values

    

    if pre_args.train != None:
        tools.entrainement_models(pre_args, args, x_train, y_train)
    
    
    if pre_args.model != None:
        print(x_train.shape[1], x_test.shape[1] ) 
        rslt = tools.exec_models(pre_args, args, x_test)
        print(x_train.shape[1], x_test.shape[1] ) 
            
        # construction des resultats
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
            filename = f"{args.m_custom_output}matrice_confusion_{suffix}.json"
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
                n_train=len(y_train),
                n_test=len(x_test), 
                dossier_fichier= args.m_custom_output
            )




if __name__ == "__main__":
    main()
