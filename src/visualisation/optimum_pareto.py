import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def construire_optimum(knn_csv, svm_csv, lr_csv, rf_csv):

    # Algo
    files = {
        'KNN': knn_csv,
        'SVM': svm_csv,
        'LR': lr_csv,
        'RF': rf_csv
    }

    algos = []
    precisions = []
    temps_total = []

    for algo, file in files.items():
        df = pd.read_csv(file)
        precision = df['precision'].iloc[0]
        temps_entrain = df['temps_entrainement'].iloc[0]
        temps_test = df['temps_test'].iloc[0]

        temps = temps_entrain + temps_test  # Somme des temps d'entraînement et de test
        algos.append(algo)
        precisions.append(precision)
        temps_total.append(temps)

    precisions = np.array(precisions)
    temps_total = np.array(temps_total)

    couleurs = ['blue', 'brown', 'orange', 'green']
    annotations = ['Knn', 'Svm', 'Lr', 'Rf']

    fig, ax = plt.subplots(figsize=(12, 7))

    for i in range(len(algos)):
        ax.scatter(precisions[i], temps_total[i], color=couleurs[i], marker='s', s=130, label=algos[i])
        ax.text(precisions[i] + 0.003, temps_total[i] + 2, annotations[i], fontsize=11)

    # Tri
    sorted_indices = np.argsort(precisions)
    x_sorted = precisions[sorted_indices]
    y_sorted = temps_total[sorted_indices]

    # Cubic
    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
    spline = make_interp_spline(x_sorted, y_sorted, k=3)
    y_smooth = spline(x_smooth)
    ax.plot(x_smooth, y_smooth, color='gray', linestyle='--', linewidth=1.5, label='Courbe de tendance')

    # Optimum de Pareto
    def get_pareto_optimum(x, y):
        sorted_idx = np.argsort(x)
        pareto = [sorted_idx[0]]
        for i in sorted_idx[1:]:
            if y[i] <= y[pareto[-1]]:
                pareto.append(i)
        return pareto

    pareto_idx = get_pareto_optimum(precisions, temps_total)
    ax.plot(precisions[pareto_idx], temps_total[pareto_idx], color='red', linewidth=2.5, label='Front de Pareto')

    ax.set_xlabel("Précision", fontsize=12)
    ax.set_ylabel("Temps total d'exécution (s)", fontsize=12)
    ax.set_title("Optimum de Pareto : Précision En Fonction de  Temps Total(entrainement+test)", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()

    def a_la_fermeture(pareto):
        print("\n=== Données affichées ===")
        for i in range(len(algos)):
            print(f"{algos[i]} : précision = {precisions[i]:.4f}, temps total = {temps_total[i]:.2f} s")

    fig.canvas.mpl_connect('close_event', a_la_fermeture)

    plt.show()

