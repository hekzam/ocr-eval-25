import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

with open('matrice_confusion_svm.json') as f:
    svm_cm = np.array(json.load(f))

plt.figure(figsize=(12, 10))
sns.heatmap(svm_cm,
           annot=True,
           fmt='d',
           cmap='Reds',
           cbar_kws={'label': 'Nombre de prédictions'},
           linewidths=0.5,
           linecolor='lightgray',
           xticklabels=range(10),
           yticklabels=range(10))

plt.title('Matrice de Confusion - SVM', pad=20, fontsize=16)
plt.xlabel('Classes Prédites', fontsize=12)
plt.ylabel('Classes Réelles', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('matrice_confusion_svm.png', dpi=300)
plt.show()