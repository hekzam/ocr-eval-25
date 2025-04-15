import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
import os


# # === Vérification du nombre de lignes dans les fichiers
# with open("resources/paths_mnist.txt", "r") as f:
#     image_paths = f.read().splitlines()
#     print(f"🖼️ Nombre d'images : {len(image_paths)}")

# with open("resources/mnist_label.txt", "r") as f:
#     labels = f.read().splitlines()
#     print(f"🏷️ Nombre de labels : {len(labels)}")





# === Chargement des paths et labels
with open("resources/paths_mnist.txt", "r") as f:
    image_paths = f.read().splitlines()

with open("resources/mnist_label.txt", "r") as f:
    labels = [int(line.strip()) for line in f.readlines()]

assert len(image_paths) == len(labels), "❌ Le nombre d'images ne correspond pas aux labels !"

print(f"📥 Chargement de {len(image_paths)} images...")

X = []
y = []

for path, label in zip(image_paths, labels):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Image manquante : {path}")
        continue

    img = img.astype("float32") / 255.0
    X.append(img.flatten())  # convertit l’image 28x28 → vecteur de 784
    y.append(label)

X = np.array(X)
y = np.array(y)

# === Split train/test
X = X[:10000]
y = y[:10000]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Entraînement SVM
print("⚙️ Entraînement du modèle SVM...")
clf = svm.SVC(probability=True)
clf.fit(X_train, y_train)

# === Évaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Précision du modèle SVM : {acc:.4f}")

# === Sauvegarde


os.makedirs("models", exist_ok=True)
dump(clf, "models/svm_digit_model.joblib")
print("💾 Modèle sauvegardé dans : models/svm_digit_model.joblib")


print("Exemple de label/image :")
print(f"→ {image_paths[0]} --> label = {labels[0]}")

print("Labels uniques :", sorted(set(labels)))


import matplotlib.pyplot as plt
import cv2

with open("resources/paths_mnist.txt", "r") as f:
    image_paths = f.read().splitlines()

with open("resources/mnist_label.txt", "r") as f:
    labels = [int(line.strip()) for line in f.readlines()]

for i in range(10):
    img = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)
    plt.subplot(2, 5, i+1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {labels[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
