import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

knn = pd.read_csv('resultats_knn.csv')
lr = pd.read_csv('resultats_logistic_regression.csv')
rf = pd.read_csv('resultats_random_forest.csv')
svm = pd.read_csv('resultats_svm.csv')

algorithmes = ['KNN', 'Régression Logistique', 'Random Forest', 'SVM']
couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8B4513']  # Bleu, Orange, Vert, Marron

# Calculer l'intervalle de confiance
def intervalle_confiance(donnees, confiance=0.95):
    n = len(donnees)
    moyenne = np.mean(donnees)
    erreur_standard = stats.sem(donnees)
    marge_erreur = erreur_standard * stats.t.ppf((1 + confiance) / 2, n - 1)
    return moyenne, marge_erreur

# listes
datasets = [knn, lr, rf, svm]
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

x = np.arange(len(algorithmes))
largeur = 0.6

# Temps d'entraînement
bars_train = ax1.bar(x, temps_train, yerr=err_train, capsize=12, color=couleurs,
                     edgecolor='black', width=largeur)
ax1.set_xticks(x)
ax1.set_xticklabels(algorithmes, rotation=30, fontsize=18)
ax1.set_ylabel('Temps (secondes)', fontsize=20)
ax1.set_title("Temps moyen d'entraînement (± IC 95%)", fontsize=24)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
max_train = max([t + e for t, e in zip(temps_train, err_train)]) * 1.3
ax1.set_ylim(0, max_train)
for i, (val, err) in enumerate(zip(temps_train, err_train)):
    txt = f"{val:.2f} ± {err:.2f}" if err > 0 else f"{val:.2f}"
    ax1.text(i, val + max_train * 0.03, txt, ha='center', fontsize=16, fontweight='bold')

# Temps de test
bars_test = ax2.bar(x, temps_test, yerr=err_test, capsize=12, color=couleurs,
                    edgecolor='black', width=largeur)
ax2.set_xticks(x)
ax2.set_xticklabels(algorithmes, rotation=30, fontsize=18)
ax2.set_title("Temps moyen de test (± IC 95%)", fontsize=24)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
max_test = max([t + e for t, e in zip(temps_test, err_test)]) * 1.3
ax2.set_ylim(0, max_test)
for i, (val, err) in enumerate(zip(temps_test, err_test)):
    txt = f"{val:.2f} ± {err:.2f}" if err > 0 else f"{val:.2f}"
    ax2.text(i, val + max_test * 0.03, txt, ha='center', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()

# --- Calcul des performances ---
precision = [df['precision'].mean() for df in datasets]
accuracy = [df['accuracy'].mean() for df in datasets]
f1_score = [df['f1_score'].mean() for df in datasets]
recall = [df['recall'].mean() for df in datasets]
balanced = [df['balanced_accuracy'].mean() for df in datasets]
temps_total = [round(t + ts, 2) for t, ts in zip(temps_train, temps_test)]

# --- Résumé  ---
resultats = pd.DataFrame({
    'Algorithme': algorithmes,
    'Tps Entrain (s)': [round(t, 2) for t in temps_train],
    'IC Entrain (±s)': [round(e, 2) for e in err_train],
    'Tps Test (s)': [round(t, 2) for t in temps_test],
    'IC Test (±s)': [round(e, 2) for e in err_test],
    'Précision': [round(p, 4) for p in precision],
    'Accuracy': [round(a, 4) for a in accuracy],
    'F1-score': [round(f1, 4) for f1 in f1_score],
    'Recall': [round(r, 4) for r in recall],
    'Balanced Acc.': [round(b, 4) for b in balanced],
    'Tps Total (s)': temps_total
})


print("\n RÉCAPITULATION DES PERFORMANCES (IC = Intervalle de Confiance à 95%) :\n")
print(resultats.to_string(index=False, float_format="{:.4f}".format))
