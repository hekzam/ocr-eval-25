import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from utils.extraction_features import zoning_features, extract_4lrp_features
import csv
import argparse


# lit les paths et labels et retourne un tuple
def read_paths_and_labels(path_file, label_file):
    
    with open(path_file, "r") as f:
        paths = [line.strip() for line in f]
    with open(label_file, "r") as f:
        labels = [int(line.strip()) for line in f]
    return list(zip(paths, labels))


# Prétraitement d'images
def process_image(image_path, label, feature_type="flatten+zoning"):
    img = Image.open(image_path).convert('L').resize((28, 28)) # convert en niveaux de gris + dimension MNIST
    img_array = np.array(img)
    features = []

    if "flatten" in feature_type:
        flatten = img_array.flatten() / 255.0 # normalisation [0, 1]
        features.extend(flatten.tolist())

    if "zoning" in feature_type:
        zoning = zoning_features(img_array, n_zones=4) # 4x4 features de zoning
        features.extend(zoning)

    if "4lrp" in feature_type:
        lrp = extract_4lrp_features(img_array, n_zones=4) # méthode 4 level resolution pyramid
        features.extend(lrp)

    return [label] + features



# Applique le prétraitement à la liste d'images
def process_dataset(dataset, data_size, feature_type):
    dataset = dataset[:data_size]
    return [process_image(path, label, feature_type=feature_type) for path, label in dataset]


# Création de dataframe
def build_dataframe(data, feature_type, n_zones=4):
    n_features = 0
    columns = ['label']

    if "flatten" in feature_type:
        columns += [f'pixel{i}' for i in range(28*28)]
        n_features += 784 # pixel 0 à pixel 783
    if "zoning" in feature_type:
        columns += [f'zone_{i}' for i in range(n_zones * n_zones)]
        n_features += n_zones * n_zones # zone 0 à zone 15
    if "4lrp" in feature_type:
        columns += [f'lrp_{i}' for i in range(4 * n_zones * n_zones)] 
        n_features += 4 * n_zones * n_zones

    # Vérification que les longueurs correspondent
    expected_cols = 1 + n_features
    for i, row in enumerate(data):
        if len(row) != expected_cols:
            raise ValueError(f"Ligne {i} a {len(row)} valeurs, attendu {expected_cols}")

    return pd.DataFrame(data, columns=columns)


# par défaut il utilise la méthode flatten + zoning
if __name__ == "__main__":

    feature_type = "flatten+zoning"

    # Fichiers
    mnist_path_file = "resources/paths_mnist.txt"
    mnist_label_file = "resources/mnist_label.txt"
    custom_path_file = "resources/paths_custom.txt"
    custom_label_file = "resources/custom_label.txt"

    # Sorties
    parquet_train_path = "resources/data_utilisees/train_data.parquet"
    parquet_test_path = "resources/data_utilisees/test_data.parquet"

    # Lecture des données
    mnist_data = read_paths_and_labels(mnist_path_file, mnist_label_file)
    custom_data = read_paths_and_labels(custom_path_file, custom_label_file)

    # Split 80/20 avec random_state pour reproductibilité
    mnist_train, mnist_test = train_test_split(mnist_data, test_size=0.2, random_state=42, stratify=[label for _, label in mnist_data])
    custom_train, custom_test = train_test_split(custom_data, test_size=0.2, random_state=42, stratify=[label for _, label in custom_data])

    print(f"MNIST : {len(mnist_train)} train / {len(mnist_test)} test")
    print(f"Custom : {len(custom_train)} train / {len(custom_test)} test")

    # Traitement des images
    with open(mnist_label_file) as f:
        count = sum(1 for _ in f)
    with open(custom_label_file) as f:
        count += sum(1 for _ in f)
    
    train_data = process_dataset(mnist_train + custom_train, int(count*0.8))
    test_data = process_dataset(mnist_test + custom_test, int(count*0.2))

    # Création des DataFrames
    df_train = build_dataframe(train_data, feature_type, n_zones=4)
    df_test = build_dataframe(test_data, feature_type, n_zones=4)


    # Sauvegarde
    df_train.to_parquet(parquet_train_path, index=False)
    df_test.to_parquet(parquet_test_path, index=False)

    print("Données sauvegardées dans :")
    print(f"  - {parquet_train_path}")
    print(f"  - {parquet_test_path}")

