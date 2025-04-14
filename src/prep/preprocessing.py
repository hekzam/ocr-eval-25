import os
import cv2
import numpy as np

def traiter_image(image_path, output_path):
    # Chargement de l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Erreur : Impossible de charger {image_path}")
        return
    # Suppression d'une bordure de 5 pixels autour de l'image
    cropped = image[5:-5, 5:-5]
    # Binarisation automatique de l'image (noir/blanc) avec la méthode d'Otsu
    _, binarized = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Inversion des couleurs (fond noir, chiffre blanc)
    inverted = 255 - binarized
    # Détection des composantes connexes (objets blancs sur fond noir)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted)
    # Filtrage des petits composants (bruit) : on ne garde que ceux avec une taille suffisante
    min_area = 30
    cleaned = np.zeros_like(inverted)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    # Recherche des coordonnées des pixels blancs restants (le chiffre)
    coords = cv2.findNonZero(cleaned)
    if coords is None:
        print(f"Aucun chiffre détecté dans {image_path}")
        return
    # Délimitation du rectangle englobant le chiffre
    x, y, w, h = cv2.boundingRect(coords)
    digit = cleaned[y:y+h, x:x+w]
    # Création d'une image carrée centrée autour du chiffre avec un padding de 10 pixels
    padding = 10
    size = max(w, h) + padding
    square = np.full((size, size), 0, dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = digit
    # Redimensionnement final à 28x28 pixels (format MNIST)
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_CUBIC)
    # Création du dossier de sortie si nécessaire, puis sauvegarde de l'image traitée
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, resized)
    print(f"✅ {output_path}")

# === Lecture du fichier contenant les paths
if __name__ == "__main__":
    txt_path = "resources/paths_custom.txt"       # chemin du .txt généré
    output_root = "resources/custom"           # dossier de sortie
    # Lecture des chemins depuis le fichier
    with open(txt_path, 'r') as f:
        paths = f.read().splitlines()
    # Traitement uniquement des images contenant un chiffre (on ignore les cases cochées ou vides)
    for image_path in paths:
        if "digit" not in os.path.basename(image_path):
            continue  # ignore les cases cochées ou vides

        filename = os.path.basename(image_path)
        output_path = os.path.join(output_root, filename)
        # Application du prétraitement à l'image
        traiter_image(image_path, output_path)


