import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def construire_optimum(knn_csv, svm_csv, lr_csv, rf_csv):
    def meilleure_ligne(csv_file):
        df = pd.read_csv(csv_file)
        best_idx = df['precision'].idxmax()
        return df.loc[best_idx]

    best_knn = meilleure_ligne(knn_csv)
    best_lr = meilleure_ligne(lr_csv)
    best_rf = meilleure_ligne(rf_csv)
    best_svm = meilleure_ligne(svm_csv)

    algos = ['KNN', 'LR', 'RF']
    lignes = [best_knn, best_lr, best_rf]

    print("===Résultat par algorithme (KNN, LR, RF, SVM) ===")
    entetes = ['Algo', 'n_train', 'n_test', 'Tps entrainement (s)', 'Tps test (s)', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'Balanced_accuracy']
    print(" | ".join(f"{h:<18}" for h in entetes))
    print("-" * 130)

    precisions = []
    temps_total = []
    for algo, row in zip(algos, lignes):
        n_train = int(row.get('n_train', np.nan))
        n_test = int(row.get('n_test', np.nan))
        tps_entrain = float(row.get('temps_entrainement', np.nan))
        tps_test = float(row.get('temps_test', np.nan))
        accuracy = float(row.get('accuracy', np.nan))
        precision = float(row.get('precision', np.nan))
        recall = float(row.get('recall', np.nan))
        f1_score = float(row.get('f1_score', np.nan))
        balanced_accuracy = float(row.get('balanced_accuracy', np.nan))

        precisions.append(precision)
        temps_total.append(tps_entrain + tps_test)

        print(f"{algo:<18} | "
              f"{n_train:<18} | "
              f"{n_test:<18} | "
              f"{tps_entrain:<18.4f} | "
              f"{tps_test:<15.4f} | "
              f"{accuracy:<8.4f} | "
              f"{precision:<9.4f} | "
              f"{recall:<7.4f} | "
              f"{f1_score:<8.4f} | "
              f"{balanced_accuracy:<17.4f}")

    # SVM détails
    n_train = int(best_svm.get('n_train', np.nan))
    n_test = int(best_svm.get('n_test', np.nan))
    tps_entrain = float(best_svm.get('temps_entrainement', np.nan))
    tps_test = float(best_svm.get('temps_test', np.nan))
    accuracy = float(best_svm.get('accuracy', np.nan))
    precision = float(best_svm.get('precision', np.nan))
    recall = float(best_svm.get('recall', np.nan))
    f1_score = float(best_svm.get('f1_score', np.nan))
    balanced_accuracy = float(best_svm.get('balanced_accuracy', np.nan))

    print(f"{'SVM':<18} | "
          f"{n_train:<18} | "
          f"{n_test:<18} | "
          f"{tps_entrain:<18.4f} | "
          f"{tps_test:<15.4f} | "
          f"{accuracy:<8.4f} | "
          f"{precision:<9.4f} | "
          f"{recall:<7.4f} | "
          f"{f1_score:<8.4f} | "
          f"{balanced_accuracy:<17.4f}")

    precisions = np.array(precisions)
    temps_total = np.array(temps_total)

    # Graphe KNN, LR, RF
    fig, ax = plt.subplots(figsize=(12, 7))

    couleurs = ['blue', 'orange', 'green']
    annotations = ['KNN', 'LR', 'RF']

    for i in range(len(algos)):
        ax.scatter(precisions[i], temps_total[i], color=couleurs[i], marker='s', s=130)
        ax.text(precisions[i] + 0.002, temps_total[i] + 0.5, annotations[i], fontsize=12)

    ordre = np.argsort(precisions)
    ax.plot(precisions[ordre], temps_total[ordre], linestyle='-', color='black', linewidth=2, label='Courbe (ordre précision)')

    ax.set_xlabel("Précision", fontsize=13)
    ax.set_ylabel("Temps total d'exécution (s)", fontsize=13)
    ax.set_title("Courbe résultat pour KNN, LR, RF ", fontsize=15)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()

    # Graphe SVM
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    temps_svm = tps_entrain + tps_test


    ax2.plot([precision - 0.01, precision + 0.01], [temps_svm - 1, temps_svm + 1], color='brown', linewidth=2, label='Courbe SVM')
    ax2.scatter(precision, temps_svm, color='brown', marker='s', s=130, label='SVM')
    ax2.text(precision + 0.002, temps_svm + 0.5, 'SVM', fontsize=12)
    ax2.set_xlabel("Précision", fontsize=12)
    ax2.set_ylabel("Temps total d'exécution (s)", fontsize=12)
    ax2.set_title("Courbe Svm", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    plt.tight_layout()

    plt.show()


