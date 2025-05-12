import os
import cv2
import numpy as np

def traiter_image(image_path, output_path):
    # Chargement de l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Erreur : Impossible de charger {image_path}")
        return False
    # Suppression d'une bordure de 5 pixels autour de l'image
    cropped = image[5:-5, 5:-5]

    # Binarisation avec Otsu
    _, binarized = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Inversion des couleurs (fond noir, chiffre blanc)
    inverted = 255 - binarized

    # Composantes connexes (objets blancs sur fond noir)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted)

    # Nettoyage bruit : on ne garde que ceux avec une taille suffisante
    min_area = 10
    cleaned = np.zeros_like(inverted)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    # Recherche des coordonnées des pixels blancs restants (le chiffre)
    coords = cv2.findNonZero(cleaned)
    if coords is None:
        return False
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

    # Redimensionnement final ( format mnist)
    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_CUBIC)

    # Création du dossier de sortie si nécessaire, puis sauvegarde de l'image traitée
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, resized)
    print(f"✅ {output_path}")
    return True

# === Lecture du fichier contenant les paths
if __name__ == "__main__":
    input_file = "resources/paths_parsed.txt"
    output_folder = "resources/custom"
    # Lecture des chemins depuis le fichier
    with open(input_file, 'r') as f:
        paths = f.read().splitlines()

    total = len(paths)
    success = 0
    failed = 0
    # Traitement uniquement des images contenant un chiffre (on ignore les cases cochées ou vides)
    for image_path in paths:
        # Exemple : .../parsed-200dpi/3/subimg/raw-0-digit.0.93.png
        parent_folder = os.path.basename(os.path.dirname(os.path.dirname(image_path)))  # ex: "3"
        filename = os.path.basename(image_path)  # ex: raw-0-digit.0.93.png

        # format correct : raw-3-digit.0.93.png (on remplace le "0" par le vrai dossier d'origine)
        parts = filename.split("-")
        parts[1] = parent_folder  # on injecte le bon dossier (3)
        final_filename = "-".join(parts)

        output_path = os.path.join(output_folder, final_filename)

        if traiter_image(image_path, output_path):
            success += 1
        else:
            failed += 1


    print("\n Résumé final du prétraitement :")
    print(f"🔢 Total         : {total}")
    print(f"✅ Succès        : {success}")
    print(f"⚠️  Échecs        : {failed}")
