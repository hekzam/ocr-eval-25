import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Charger la matrice de confusion 
with open('matrice_confusion_logistic_regression.json') as f:
    lr_cm = np.array(json.load(f))

#Style
plt.style.use('default')
sns.set_style("white")

#noir et blanc
black_white = sns.light_palette("black", as_cmap=True)

plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(lr_cm, 
                     annot=True, 
                     fmt='d', 
                     cmap=black_white,
                     cbar_kws={'label': 'Nombre de prédictions'},
                     linewidths=0.5,
                     linecolor='gray',
                     xticklabels=range(10), 
                     yticklabels=range(10))

plt.title('Matrice de Confusion - Régression Logistique', pad=20, fontsize=16)
plt.xlabel('Classes Prédites', fontsize=12)
plt.ylabel('Classes Réelles', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('matrice_confusion_lr.png', dpi=300)
plt.show()