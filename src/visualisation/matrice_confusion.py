import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def construire_matrice_confusion(nom_modele, input, output):

    # Charger la matrice de confusion depuis fichier
    with open(input) as f:
        cm = np.array(json.load(f))

    #Style
    plt.style.use('default')
    sns.set_style("white")

    match nom_modele:
        case "svm":
            color= 'Reds'
        case "random forest":
            color= sns.light_palette("green", as_cmap=True)
        case "logistic regression":
            color= sns.light_palette("black", as_cmap=True)
        case "knn":
            color= 'Blues'


    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(cm, 
                        annot=True, 
                        fmt='d', 
                        cmap=color,
                        cbar_kws={'label': 'Nombre de prédictions'},
                        linewidths=0.5,
                        linecolor='gray',
                        xticklabels=range(10), 
                        yticklabels=range(10))

    plt.title('Matrice de Confusion - '+nom_modele, pad=20, fontsize=16)
    plt.xlabel('Classes Prédites', fontsize=12)
    plt.ylabel('Classes Réelles', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output+'matrice_confusion_'+nom_modele+'.png', dpi=300)
    plt.show()