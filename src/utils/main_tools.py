import time
import json

import algorithm.knn as knn
import algorithm.logistic_regression as lr
import algorithm.random_forest as rf
import algorithm.svm as svm
import algorithm.linear_svm as linear_svm
import utils.preparation_donnees as prep_d




def calcul_model(nom_algo, module, x_test, input):
    print(f"Prédiction avec {nom_algo}...")
    start = time.perf_counter()
    y_pred = module.predict(x_test, input)
    stop = time.perf_counter()
    print(f"Prédiction terminée en {stop - start:.3f}s")
    return [nom_algo, y_pred, stop - start]



def construction_donnees(args):
    feature_type = args.r_features

    # Fichiers
    mnist_path_file = "resources/paths_mnist.txt"
    mnist_label_file = "resources/mnist_label.txt"
    custom_path_file = "resources/paths_custom.txt"
    custom_label_file = "resources/custom_label.txt"

    # Sorties
    parquet_train_path = "resources/data_utilisees/train_data.parquet"
    parquet_test_path = "resources/data_utilisees/test_data.parquet"



    # Lecture des données
    mnist_data = prep_d.read_paths_and_labels(mnist_path_file, mnist_label_file)
    custom_data = prep_d.read_paths_and_labels(custom_path_file, custom_label_file)

    # Split 80/20 avec random_state pour reproductibilité
    mnist_train, mnist_test = prep_d.train_test_split(mnist_data, test_size=1-args.r_training_ratio, random_state=42, stratify=[label for _, label in mnist_data])
    custom_train, custom_test = prep_d.train_test_split(custom_data, test_size=1-args.r_training_ratio, random_state=42, stratify=[label for _, label in custom_data])

    print(f"MNIST : {len(mnist_train)} train / {len(mnist_test)} test")
    print(f"Custom : {len(custom_train)} train / {len(custom_test)} test")

    # Traitement des images
    train_data = prep_d.process_dataset(mnist_train + custom_train, int(args.r_data_size*args.r_training_ratio), args.r_features)
    test_data = prep_d.process_dataset(mnist_test + custom_test, int(args.r_data_size*(1-args.r_training_ratio)), args.r_features)

    # Création des DataFrames
    df_train = prep_d.build_dataframe(train_data, feature_type, n_zones=4)
    df_test = prep_d.build_dataframe(test_data, feature_type, n_zones=4)

    # Sauvegarde
    df_train.to_parquet(parquet_train_path, index=False)
    df_test.to_parquet(parquet_test_path, index=False)



def entrainement_models(pre_args, args, x_train, y_train):
    #t_custom_output "linear_svm"
    if 'knn' in pre_args.train:
        nom = "knn"
        start = time.perf_counter()
        knn.train(x_train, y_train, args.t_custom_output+ "knn_model.pkl", args.knn_subset)
        stop = time.perf_counter()
        print(f"Modèle '{nom}' entraîné en {stop - start:.3f}s")

        # Sauvegarde temps dans fichier JSON
        suffix = nom.lower().replace(" ", "_")
        with open(f"results/temps_entrainement/temps_train_{suffix}.json", "w") as f:
            json.dump({"temps": stop - start}, f)
    
    if 'svm' in pre_args.train:
        nom="svm"
        start = time.perf_counter()
        svm.train(x_train, y_train, args.t_custom_output+ "svm_model.pkl", args.svm_subset, args.svm_regulation_parameter)
        stop = time.perf_counter()
        print(f"Modèle '{nom}' entraîné en {stop - start:.3f}s")

        # Sauvegarde temps dans fichier JSON
        suffix = nom.lower().replace(" ", "_")
        with open(f"results/temps_entrainement/temps_train_{suffix}.json", "w") as f:
            json.dump({"temps": stop - start}, f)

    if 'rf' in pre_args.train:
        nom = "random forest"
        start = time.perf_counter()
        rf.train(x_train, y_train, args.t_custom_output+ "rf_model.pkl", args.rf_subset)
        stop = time.perf_counter()
        print(f"Modèle '{nom}' entraîné en {stop - start:.3f}s")

        # Sauvegarde temps dans fichier JSON
        suffix = nom.lower().replace(" ", "_")
        with open(f"results/temps_entrainement/temps_train_{suffix}.json", "w") as f:
            json.dump({"temps": stop - start}, f)

    if 'lr' in pre_args.train:
        nom = "logistic regression"
        start = time.perf_counter()
        lr.train(x_train, y_train, args.t_custom_output+ "lr_model.pkl", args.lr_subset)
        stop = time.perf_counter()
        print(f"Modèle '{nom}' entraîné en {stop - start:.3f}s")

        # Sauvegarde temps dans fichier JSON
        suffix = nom.lower().replace(" ", "_")
        with open(f"results/temps_entrainement/temps_train_{suffix}.json", "w") as f:
            json.dump({"temps": stop - start}, f)


def exec_models(pre_args, args, x_test):
    rslt = []
    if 'knn' in pre_args.test:
        r = calcul_model("knn", knn, x_test, args.m_custom_input+ "knn_model.pkl")
        rslt.append(r)
    if 'svm' in pre_args.test:
        r = calcul_model("svm", svm, x_test, args.m_custom_input+ "svm_model.pkl")
        rslt.append(r)
    if 'rf' in pre_args.test:
        r = calcul_model("random forest", rf, x_test, args.m_custom_input+ "rf_model.pkl")
        rslt.append(r)
    if 'lr' in pre_args.test:
        r = calcul_model("logistic regression", lr, x_test, args.m_custom_input+ "lr_model.pkl")
        rslt.append(r) 

    return rslt
        