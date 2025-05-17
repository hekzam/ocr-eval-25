import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


knn = pd.read_csv('resultats_knn.csv').iloc[0]
lr = pd.read_csv('resultats_logistic_regression.csv')
rf = pd.read_csv('resultats_random_forest.csv').iloc[0]


algorithmes = ['KNN', 'Régression Logistique', 'Random_Forest']
couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Bleu, Orange, Vert

# Calcul des intervalles de confiance
def intervalle_confiance(donnees, confiance=0.95):
    n = len(donnees)
    moyenne = np.mean(donnees)
    erreur_standard = stats.sem(donnees)
    marge_erreur = erreur_standard * stats.t.ppf((1 + confiance) / 2, n - 1)
    return moyenne, marge_erreur

# Traitement des données
moy_train_lr, err_train_lr = intervalle_confiance(lr['temps_entrainement'])
moy_test_lr, err_test_lr = intervalle_confiance(lr['temps_test'])
precision_lr = lr['precision'].mean()

donnees_entrainement = {
    'moyennes': [knn['temps_entrainement'], moy_train_lr, rf['temps_entrainement']],
    'erreurs': [0, err_train_lr, 0]
}

donnees_test = {
    'moyennes': [knn['temps_test'], moy_test_lr, rf['temps_test']],
    'erreurs': [0, err_test_lr, 0]
}

precision = [knn['precision'], precision_lr, rf['precision']]
temps_total = [
    knn['temps_entrainement'] + knn['temps_test'],
    moy_train_lr + moy_test_lr,
    rf['temps_entrainement'] + rf['temps_test']
]

#Histogrammes
plt.figure(figsize=(14, 6))

# Histogramme des temps d'entraînement
plt.subplot(1, 2, 1)
barres = plt.bar(algorithmes, donnees_entrainement['moyennes'],
                yerr=donnees_entrainement['erreurs'],
                capsize=10,
                color=couleurs)

for barre in barres:
    hauteur = barre.get_height()
    plt.text(barre.get_x() + barre.get_width()/2, hauteur,
             f'{hauteur:.2f}s',
             ha='center', va='bottom')

plt.title('Temps moyen d\'entraînement', fontsize=14)
plt.ylabel('Temps (secondes)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Histogramme des temps de test
plt.subplot(1, 2, 2)
barres = plt.bar(algorithmes, donnees_test['moyennes'],
                yerr=donnees_test['erreurs'],
                capsize=10,
                color=couleurs)

for barre in barres:
    hauteur = barre.get_height()
    plt.text(barre.get_x() + barre.get_width()/2, hauteur,
             f'{hauteur:.2f}s',
             ha='center', va='bottom')

plt.title('Temps moyen de test', fontsize=14)
plt.ylabel('Temps (secondes)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()



# récapitulation
print("\n RÉCAPITULATION DES PERFORMANCES :")
print("(IC = Intervalle de Confiance à 95%)")

resultats = pd.DataFrame({
    'Algorithme': algorithmes,
    'Temps Entrainement (s)': donnees_entrainement['moyennes'],
    'IC Entrainement (±s)': donnees_entrainement['erreurs'],
    'Temps Test (s)': donnees_test['moyennes'],
    'IC Test (±s)': donnees_test['erreurs'], # intervalle de confience
    'Précision': precision,
    'Temps Total (s)': temps_total
})

print(resultats.round(2).to_string(index=False))