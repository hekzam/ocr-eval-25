import csv
import os
def sauvegarder_resultats_csv(
    nom_algo,
    metriques: dict,
    temps_train: float,
    temps_test: float,
    n_train: int,
    n_test: int,
    dossier_fichier="results/results_algo"
):
    """
    Sauvegarde les métriques d'un modèle dans un fichier CSV propre à l'algorithme.

    Paramètres :
    - nom_algo : str, nom de l'algorithme
    - metriques : dict, contient accuracy, precision, recall, f1_score, balanced_accuracy
    - temps_train : float, durée de l'entraînement en secondes
    - temps_test : float, durée du test en secondes
    - n_train : int, nombre de données d'entraînement
    - n_test : int, nombre de données de test
    - dossier_fichier : str, dossier où enregistrer le fichier CSV
    """

    entetes = [
        "algorithme", "n_train", "n_test",
        "temps_entrainement", "temps_test",
        "accuracy", "precision", "recall",
        "f1_score", "balanced_accuracy"
    ]

    ligne = {
        "algorithme": nom_algo,
        "n_train": n_train,
        "n_test": n_test,
        "temps_entrainement": round(temps_train, 4),
        "temps_test": round(temps_test, 4),
        "accuracy": round(metriques.get("accuracy", 0), 4),
        "precision": round(metriques.get("precision", 0), 4),
        "recall": round(metriques.get("recall", 0), 4),
        "f1_score": round(metriques.get("f1_score", 0), 4),
        "balanced_accuracy": round(metriques.get("balanced_accuracy", 0), 4),
    }

    # Chemin du fichier propre à l'algo
    fichier_csv = os.path.join(dossier_fichier, f"resultats_{nom_algo.lower().replace(' ', '_')}.csv")

    os.makedirs(os.path.dirname(fichier_csv), exist_ok=True)
    fichier_existe = os.path.exists(fichier_csv)

    with open(fichier_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=entetes)
        writer.writeheader()
        writer.writerow(ligne)

    print(f"\nRésultats enregistrés dans : {fichier_csv}")
