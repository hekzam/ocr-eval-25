import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def construire_optimum(knn_csv, lr_csv, rf_csv, output):

    # Algo
    files = {
        'KNN': knn_csv,
        'LR': lr_csv,
        'RF': rf_csv
    }

    algos = []
    precisions = []
    temps_total = []

    for algo, file in files.items():
        df = pd.read_csv(file)
        precision = df['precision'].iloc[0]
        temps_test = df['temps_test'].iloc[0]

        algos.append(algo)
        precisions.append(precision)
        temps_total.append(temps_test)

    precisions = np.array(precisions)
    temps_total = np.array(temps_total)

    couleurs = ['blue', 'orange', 'green']
    annotations = ['Knn', 'Lr', 'Rf']

    fig, ax = plt.subplots(figsize=(12, 7))

    for i in range(len(algos)):
        ax.scatter(temps_total[i], precisions[i], color=couleurs[i], marker='s', s=130, label=algos[i])
        ax.text(temps_total[i]+0.05 , precisions[i], annotations[i], fontsize=11)


    # Optimum de Pareto
    def get_pareto_optimum(x, y):
        sorted_idx = np.argsort(x)
        pareto = [sorted_idx[0]]
        for i in sorted_idx[1:]:
            if y[i] >= y[pareto[-1]]:
                pareto.append(i)
        return pareto

    pareto_idx = get_pareto_optimum(temps_total, precisions)
    ax.plot(temps_total[pareto_idx], precisions[pareto_idx], color='red', linewidth=2.5, label='Front de Pareto')

    ax.set_xlabel("Temps d'exécution (s)", fontsize=12)
    ax.set_ylabel("Précision", fontsize=12)
    ax.set_title("Optimum de Pareto : Précision En Fonction de  Temps Test", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()

    def a_la_fermeture(pareto):
        print("\n=== Données affichées ===")
        for i in range(len(algos)):
            print(f"{algos[i]} : temps total = {temps_total[i]:.2f}, précision = {precisions[i]:.4f} s")

    fig.canvas.mpl_connect('close_event', a_la_fermeture)

    plt.savefig(output+'\optimum_pareto.png', dpi=300)
    plt.show()