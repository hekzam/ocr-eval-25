import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Charger la matrice de confusion KNN
with open('matrice_confusion_knn.json') as f:
    knn_cm = np.array(json.load(f))

#Style
plt.style.use('default')
sns.set_style("whitegrid")

#  KNN
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(knn_cm, 
                     annot=True, 
                     fmt='d', 
                     cmap='Blues',
                     cbar_kws={'label': 'Nombre de prédictions'},
                     linewidths=0.5,
                     linecolor='lightgray',
                     xticklabels=range(10), 
                     yticklabels=range(10))

plt.title('Matrice de Confusion - KNN', pad=20, fontsize=16)
plt.xlabel('Classes Prédites', fontsize=12)
plt.ylabel('Classes Réelles', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('matrice_confusion_knn.png', dpi=300)
plt.show()