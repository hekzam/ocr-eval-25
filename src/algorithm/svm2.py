import numpy as np
import cv2
import string
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Caractères à reconnaître
letters = string.ascii_uppercase
digits = string.digits

# Dictionnaires pour stocker les données
samples = {}
responses = {}
svms = {}

# Chemins dans ta structure de projet
BASE_PATH = "ocr_eval_25/resources/"
IMG_PATHS_FILE = BASE_PATH + "paths_custom.txt"
LABELS_FILE = BASE_PATH + "custom_label.txt"

# Initialiser les SVMs
for c in letters + digits:
    svms[c] = SVC(probability=True, kernel='linear')

# Fonction pour charger les images et leurs labels
def load_data(image_paths_file, labels_file):
    with open(image_paths_file, "r") as f:
        image_paths = [line.strip() for line in f.readlines()]
    with open(labels_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return image_paths, labels

# Fonction de sélection : One-vs-All
def oneAgainstAll(results):
    results = sorted(results, key=lambda t: t[1], reverse=True)
    return results[0][0]

# Entraînement
def train():
    image_paths, labels = load_data(IMG_PATHS_FILE, LABELS_FILE)

    for c in letters + digits:
        samples[c] = []
        responses[c] = []

    for img_path, label in zip(image_paths, labels):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (24, 32))  # Standardisation
        hist = hog(img, block_norm='L2-Hys')

        for c in letters + digits:
            response = 1 if c == label else -1
            samples[c].append(hist)
            responses[c].append(response)

    print("Entraînement en cours...")
    for c in letters + digits:
        X = np.float32(samples[c])
        y = np.array(responses[c])
        svms[c].fit(X, y)
    print("Entraînement terminé.")

# Test
def test():
    image_paths, labels = load_data(IMG_PATHS_FILE, LABELS_FILE)

    n_tests = 0
    n_errors = 0
    confusion = np.zeros((len(letters + digits), len(letters + digits)), dtype=int)
    conf_index = {c: i for i, c in enumerate(letters + digits)}

    for img_path, label in zip(image_paths, labels):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (24, 32))
        hist = hog(img, block_norm='L2-Hys')
        col = letters if label in letters else digits
        results = []

        for c in col:
            prob = svms[c].predict_proba([hist])[0][1]
            results.append((c, prob))

        predicted = oneAgainstAll(results)
        confusion[conf_index[label]][conf_index[predicted]] += 1
        n_tests += 1
        if predicted != label:
            n_errors += 1

    print(f"{n_tests} tests. {n_errors} erreurs. Taux d'erreur : {n_errors / n_tests:.2%}")

    # Matrice de confusion
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion, interpolation='nearest', cmap='Blues')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(letters + digits)))
    ax.set_yticks(np.arange(len(letters + digits)))
    ax.set_xticklabels(list(letters + digits))
    ax.set_yticklabels(list(letters + digits))
    plt.xticks(rotation=90)
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion")
    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    train()
    test()
