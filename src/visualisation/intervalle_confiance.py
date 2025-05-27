import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def construire_intervalle_confiance(knn_csv, lr_csv, rf_csv, output):
    algorithmes = ['KNN', 'Régression Logistique', 'Random Forest']
    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Bleu, Orange, Vert

    # Calculer l'intervalle de confiance
    def intervalle_confiance(donnees, confiance=0.95):
        n = len(donnees)
        moyenne = np.mean(donnees)
        erreur_standard = stats.sem(donnees)
        marge_erreur = erreur_standard * stats.t.ppf((1 + confiance) / 2, n - 1)
        return moyenne, marge_erreur

    # listes
    datasets = [knn_csv, lr_csv, rf_csv]
    temps_train, temps_test = [], []
    err_train, err_test = [], []

    for data in datasets:
        if len(data) > 1:
            moy_train, err_t = intervalle_confiance(data['temps_entrainement'])
            moy_test, err_te = intervalle_confiance(data['temps_test'])
        else:
            moy_train = data['temps_entrainement'].iloc[0]
            err_t = 0
            moy_test = data['temps_test'].iloc[0]
            err_te = 0
        temps_train.append(moy_train)
        temps_test.append(moy_test)
        err_train.append(err_t)
        err_test.append(err_te)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)  # Taille augmentée

    x = np.arange(len(algorithmes))
    largeur = 0.6


    # Temps d'entraînement
    ax1.bar(x, temps_train, yerr=err_train, capsize=12, color=couleurs,
                        edgecolor='black', width=largeur)
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithmes, rotation=30, fontsize=18)  # Texte plus grand
    ax1.set_ylabel('Temps (secondes)', fontsize=20)
    ax1.set_title("Temps moyen d'entraînement (± IC 95%)", fontsize=24)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    max_train = max([t + e for t, e in zip(temps_train, err_train)]) * 1.3
    


    # Temps de test
    ax2.bar(x, temps_test, yerr=err_test, capsize=12, color=couleurs,
                        edgecolor='black', width=largeur)
    ax2.set_xticks(x)
    ax2.set_xticklabels(algorithmes, rotation=30, fontsize=18)
    ax2.set_title("Temps moyen de test (± IC 95%)", fontsize=24)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    max_test = max([t + e for t, e in zip(temps_test, err_test)]) * 1.3
    
    limite_y = max(max_train, max_test)
    ax2.set_ylim(0, limite_y)
    for i, (val, err) in enumerate(zip(temps_test, err_test)):
        txt = f"{val:.2f} ± {err:.2f}" if err > 0 else f"{val:.2f}"
        ax2.text(i, val + max_test * 0.03, txt, ha='center', fontsize=16, fontweight='bold')
    ax1.set_ylim(0, limite_y)
    for i, (val, err) in enumerate(zip(temps_train, err_train)):
        txt = f"{val:.2f} ± {err:.2f}" if err > 0 else f"{val:.2f}"
        ax1.text(i, val + max_train * 0.03, txt, ha='center', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output+'\intervalle_confiance.png', dpi=300)
    plt.show()
    