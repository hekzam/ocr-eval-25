import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Charger la matrice de confusion depuis fichier
with open('matrice_confusion_random_forest.json') as f:
    rf_cm = np.array(json.load(f))

#Style
plt.style.use('default')
sns.set_style("white")

#vert et blanc
green_white = sns.light_palette("green", as_cmap=True)

plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(rf_cm, 
                     annot=True, 
                     fmt='d', 
                     cmap=green_white,
                     cbar_kws={'label': 'Nombre de prédictions'},
                     linewidths=0.5,
                     linecolor='gray',
                     xticklabels=range(10), 
                     yticklabels=range(10))

plt.title('Matrice de Confusion - Random Forest', pad=20, fontsize=16)
plt.xlabel('Classes Prédites', fontsize=12)
plt.ylabel('Classes Réelles', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('matrice_confusion_rf.png', dpi=300)
plt.show()