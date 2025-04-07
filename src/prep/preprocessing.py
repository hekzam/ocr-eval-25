import os
import cv2
import numpy as np

def traiter_image(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"❌ Erreur : Impossible de charger {image_path}")
        return

    cropped = image[5:-5, 5:-5]
    _, binarized = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = 255 - binarized

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted)
    min_area = 30
    cleaned = np.zeros_like(inverted)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    coords = cv2.findNonZero(cleaned)
    if coords is None:
        print(f"Aucun chiffre détecté dans {image_path}")
        return
    x, y, w, h = cv2.boundingRect(coords)
    digit = cleaned[y:y+h, x:x+w]

    padding = 10
    size = max(w, h) + padding
    square = np.full((size, size), 0, dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = digit

    resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_CUBIC)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, resized)
    print(f"✅ {output_path}")

# === Lecture du fichier contenant les paths
if __name__ == "__main__":
    txt_path = "resources/paths_custom.txt"       # chemin du .txt généré
    output_root = "resources/custom"           # dossier de sortie

    with open(txt_path, 'r') as f:
        paths = f.read().splitlines()

    for image_path in paths:
        if "digit" not in os.path.basename(image_path):
            continue  # ignore les cases cochées ou vides

        filename = os.path.basename(image_path)
        output_path = os.path.join(output_root, filename)

        traiter_image(image_path, output_path)


