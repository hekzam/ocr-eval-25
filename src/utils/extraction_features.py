import numpy as np

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
