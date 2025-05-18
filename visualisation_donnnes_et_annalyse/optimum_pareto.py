import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

algos = ['KNN', 'Régression Logistique', 'Random Forest', 'SVM']
precisions = np.array([0.9603, 0.8950, 0.9510,  0.9280])
temps_total = np.array([9.2500, 29.5900, 10.1000, 103.7300])
couleurs = ['blue', 'orange', 'green', 'brown']
annotations = ['Knn', 'Lr', 'Rf', 'Svm']

plt.figure(figsize=(12, 7))

# les points
for i in range(len(algos)):
    plt.scatter(precisions[i], temps_total[i], color=couleurs[i], marker='s', s=130, label=algos[i])
    plt.text(precisions[i] + 0.003, temps_total[i] + 2, annotations[i], fontsize=11)

# Tri
sorted_indices = np.argsort(precisions)
x_sorted = precisions[sorted_indices]
y_sorted = temps_total[sorted_indices]

# cube
x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)
spline = make_interp_spline(x_sorted, y_sorted, k=3)
y_smooth = spline(x_smooth)
plt.plot(x_smooth, y_smooth, color='gray', linestyle='--', linewidth=1.5, label='Courbe de tendance')

# Optimum de Pareto
def get_pareto_optimum(x, y):
    sorted_idx = np.argsort(x)
    pareto = [sorted_idx[0]]
    for i in sorted_idx[1:]:
        if y[i] <= y[pareto[-1]]:
            pareto.append(i)
    return pareto

pareto_idx = get_pareto_optimum(precisions, temps_total)
plt.plot(precisions[pareto_idx], temps_total[pareto_idx], color='red', linewidth=2.5, label='Front de Pareto')


plt.xlabel("Précision (taux de bonne prédiction)", fontsize=12)
plt.ylabel("Temps total d'exécution (s)", fontsize=12)
plt.title("Optimum de Pareto : Précision vs Temps d'entraînement", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
