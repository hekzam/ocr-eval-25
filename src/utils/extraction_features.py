import numpy as np
from PIL import Image

def zoning_features(img_array, n_zones=4):
    """
    Découpe une image 28x28 en N x N zones et retourne un vecteur
    de densité de pixels blancs (valeurs 255) dans chaque zone.

    Parametres:
    - img_array : numpy array (28x28) en niveaux de gris
    - n_zones : nombre de zones par dimension (4 par défaut → 16 zones)

    Retourne:
    - features : liste de densité (longueur n_zones^2)
    """
    zone_size = 28 // n_zones
    features = []

    for i in range(n_zones):
        for j in range(n_zones):
            zone = img_array[i*zone_size:(i+1)*zone_size, j*zone_size:(j+1)*zone_size]
            density = np.sum(zone == 255) / (zone_size * zone_size)
            features.append(density)

    return features

def extract_4lrp_features(img_array, n_zones=4):
    """
    Extrait des features à 4 niveaux de résolution à partir d'une image 2D (img_array).
    Chaque niveau est obtenu par sous-échantillonnage progressif.
    Pour chaque niveau, on applique un zoning (densité de pixels blancs par zone).

    Paramètres :
        img_array (np.ndarray) : image 2D en niveaux de gris
        n_zones (int) : découpe l'image en n_zones x n_zones par niveau

    Retour :
        features (list) : vecteur concaténé des features de chaque niveau
    """
    features = []

    current = img_array.copy()
    for level in range(4):
        # Normaliser l'image entre 0 et 1 si ce n'est pas fait
        if current.max() > 1.0:
            current = current / 255.0

        h, w = current.shape
        zone_h = h // n_zones
        zone_w = w // n_zones

        for i in range(n_zones):
            for j in range(n_zones):
                zone = current[i*zone_h:(i+1)*zone_h, j*zone_w:(j+1)*zone_w]
                density = np.mean(zone)
                features.append(density)

        # Sous-échantillonnage (réduction de taille) pour le niveau suivant
        current = current[::2, ::2]  # divise la résolution par 2 à chaque niveau

    return features


def image_path_to_4lrp_features(image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    img_array = np.array(img)
    return extract_4lrp_features(img_array)
