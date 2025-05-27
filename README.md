# 🧠 Projet BE_25 – HEKZAM OCR

HEKZAM est une suite d’outils développée dans le but d’automatiser le traitement d’images scannées de copies d’étudiants, en ciblant particulièrement la détection et la reconnaissance de chiffres manuscrits (par exemple : numéros d’étudiant). Le projet repose sur une chaîne de traitement intégrant des techniques avancées de prétraitement d’image, de tri, d’étiquetage, ainsi que de reconnaissance automatique, s’appuyant sur des moteurs spécialisés implémentant des algorithmes d’extraction de caractéristiques et de classification.

Ce projet a également pour vocation de comparer les performances des différents algorithmes testés dans la phase de reconnaissance, tant en termes de précision que de coût computationnel. Cette démarche comparative vise à identifier le compromis entre la précision des résultats et le cout des ressources mobilisées.

---

## 📜 License

- Code: Apache-2.0
- Everything else, in particular documentation and measurements: CC-BY-SA-4.0

---

## 👥 Contributeurs

- Nourhane MALLEK
- Yennayer BENAMEUR
- Omar KARAM
- Paulin LAURENT

---

## 📁 Structure du projet
```bash
ocr_eval_25/
├── resources/
│   ├── parsed-200dpi/       → Données scannées brutes
│   ├── mnist/               → Extraits du dataset MNIST original
│   ├── custom/              → Images prétraitées prêtes pour l'entraînement
│   ├── data_utilisees       → Données prête a être utilisée par les modèles 
│   ├── paths_*.txt          → Fichiers contenant les chemins vers les images
│   ├── mnist_label.txt      → Labels associés aux images mnist
│   └── custom_label.txt     → Labels associés aux images custom
├── results/
│   ├── models/              → Stockage des models entrainés
│   ├── results_algo/        → Stockage des données produites par les models
│   ├── temps_entrainement/  → Stockage du temps d'entrainement des models et du temps d'extraction de features
│   └── visuels/             → Stockage des rendus visuels des résultats et comparaisons des models
├── src/
│   ├── prep/
│   │   ├── generate_mnist_paths.py   → Génére les paths des images mnist
│   │   ├── generate_custom_paths.py  → Génére les paths des images custom
│   │   ├── generate_custom_label.py  → Génére les labels des images custom
│   │   ├── generate_parsed_paths.py  → Génére les paths des images brutes
│   │   └── preprocessing.py          → Prétraitement des images
│   ├── algorithm/
│   │   ├── knn_search.py             → Recherche des meilleurs paramètres de knn
│   │   ├── knn.py                    → Model de reconnaissance basé sur knn
│   │   ├── linear_svm.py             → Model de reconnaissance basé sur linear svm, non inclus dans le cli car non pertinant
│   │   ├── logistic_regression.py    → Model de reconnaissance basé sur logistic regression
│   │   ├── random_forest.py          → Model de reconnaissance basé sur random forest
│   │   └── svm.py                    → Model de reconnaissance basé sur svm
│   ├── utils/
│   │   ├── extraction_features.py    → Permet de récupérer les features depuis les images
│   │   ├── main_tools.py             → Permet d'exécuter le reste du programme sans saturer hekzam.ocr
│   │   ├── preparation_donnees.py    → Transforme les images prétraitées en données utilisable par les models
│   │   └── sauvegarde_csv.py         → Sauvegarde les résultats des models en csv 
│   └── hekzam_ocr.py        → Agis en tant que main, contient le cli
└── README.txt              → Ce fichier
```
---

## ⚙️ Fonctionnalités principales

Le projet heckzam_ocr propose une suite complète d'outils pour la gestion de pipelines d'OCR basés sur des algorithmes de machine learning classiques.

Reconstruction des données

    Fusionne des jeux de données (ex : MNIST + dataset personnalisé)
    
    Extrait des caractéristiques (flatten, zoning, 4lrp)
    
    Divise automatiquement en train/test selon un ratio défini

Entraînement de modèles

    Supporte plusieurs algorithmes :
    
        KNN (K-Nearest Neighbors)
        
        SVM (Support Vector Machine)
        
        Random Forest
        
        Logistic Regression
        
        Linear SVM
        
    Possibilité de régler les sous-ensembles d'entraînement et hyperparamètres

Exécution et évaluation

    Charge les modèles entraînés et prédit sur les données de test
    
    Calcule automatiquement :
    
        Accuracy
        
        Precision
        
        Recall
        
        F1-Score
        
        Balanced Accuracy
    
    Génère une matrice de confusion par modèle
    
    Chronomètre les temps d'entraînement et d'inférence

Export et rapports
    
    Sauvegarde des résultats au format :
    
        .json (matrices, résultats bruts)
        
        .csv (résumé des performances)
    
    Structure de dossiers personnalisable
    
    Rapports compatibles avec des outils d’analyse externes

---

## 🚀 Exécution (explication cli)
Commandes générales
| Option           | Description                                                                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--rebuild_data` | Reconstruit les données à partir des fichiers label (MNIST et custom). Active les options `--r_features`, `--r_training_ratio`, `--r_data_size`. |
| `--train`        | Liste des modèles à entraîner (séparés par virgule). Valeurs autorisées : `knn`, `svm`, `rf`, `lr`, `linear_svm`.                                |
| `--model`        | Liste des modèles à exécuter sur les données de test. Identiques à ceux disponibles avec `--train`.                                              |

Options liées à --rebuild_data
| Option               | Description                                                                                                           |
| -------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `--r_features`       | Méthodes d’extraction de caractéristiques. Valeurs : `flatten`, `zoning`, `4lrp`. Combinaison possible avec virgules. |
| `--r_training_ratio` | Ratio du dataset à utiliser pour l'entraînement. Float entre `0.1` et `0.99`. Par défaut : `0.8`.                     |
| `--r_data_size`      | Nombre total d’images à utiliser (maximum = taille totale des labels). Par défaut : toutes.                           |

Options liées à --train
| Option                       | Description                                                                  |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `--t_custom_output`          | Dossier de sauvegarde des modèles entraînés. Par défaut : `results/models/`. |
| `--knn_subset`               | Nombre d'exemples utilisés pour entraîner le KNN. `-1` pour tout utiliser.   |
| `--svm_subset`               | Idem pour SVM.                                                               |
| `--svm_regulation_parameter` | Hyperparamètre de régularisation pour le SVM.                                |
| `--rf_subset`                | Idem pour Random Forest.                                                     |
| `--lr_subset`                | Idem pour Logistic Regression.                                               |

Options liées à --model
| Option              | Description                                                                                          |
| ------------------- | ---------------------------------------------------------------------------------------------------- |
| `--m_custom_input`  | Dossier contenant les modèles entraînés (à exécuter). Par défaut : `results/confusion_matrices/`.    |
| `--m_custom_output` | Dossier de sortie pour les résultats et matrices de confusion. Par défaut : `results/results_algo/`. |

Exemple execution complete par defaut:
`--rebuild_data --train knn,svm,rf,lr --test knn,svm,lr,rf --m_confusion_matrix --m_visuals_construction`

---

## 🧩 Prérequis Python

- Python 3.8 ou plus
- Dépendances listées dans le fichier `requirements.txt`

Installation recommandée :
```bash
pip install -r requirements.txt
```
