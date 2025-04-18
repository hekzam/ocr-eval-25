# 🧠 Projet BE_25 – HEKZAM OCR

HEKZAM est une suite d’outils développée dans le but d’automatiser le traitement d’images scannées de copies d’étudiants, en ciblant particulièrement la détection et la reconnaissance de chiffres manuscrits (par exemple : numéros d’étudiant). Le projet repose sur une chaîne de traitement intégrant des techniques avancées de prétraitement d’image, de tri, d’étiquetage, ainsi que de reconnaissance automatique, s’appuyant sur des moteurs spécialisés implémentant des algorithmes d’extraction de caractéristiques et de classification.

Ce projet a également pour vocation de comparer les performances des différents algorithmes testés dans la phase de reconnaissance, tant en termes de précision que de coût computationnel. Cette démarche comparative vise à identifier le compromis entre la précision des résultats et le cout des ressources mobilisées.

---

## 📜 License

- Code: Apache-2.0
- Everything else, in particular documentation and measurements: CC-BY-SA-4.0

---

## 👥 Contributeurs

- Paulin LAURENT
- Yennayer BENAMEUR
- Nourhane MALLEK
- Omar KARAM

---

## 📁 Structure du projet
```bash
ocr_eval_25/
├── resources/
│   ├── parsed-200dpi/       → Données scannées brutes
│   ├── mnist/               → Extraits du dataset MNIST original
│   ├── custom/              → Images prétraitées prêtes pour l'entraînement
│   ├── paths_*.txt          → Fichiers contenant les chemins vers les images
│   ├── mnist_label.txt      → Labels associés aux images mnist
│   └── custom_label.txt     → Labels associés aux images custom
├── src/
│   └── prep/
│       ├── generate_mnist_paths.py   → Génére les paths des images mnist
│       ├── generate_custom_paths.py  → Génére les paths des images custom
│       ├── generate_custom_label.py  → Génére les labels des images custom
│       ├── generate_parsed_paths.py  → Génére les paths des images brutes
│       └── preprocessing.py          → Prétraitement des images
├── results/                 → Résultats d’évaluation des différents modèles
└── README.txt               → Ce fichier
```
---

## ⚙️ Fonctionnalités principales

- Génération des chemins d’accès pour les datasets (`mnist`, `custom`, `parsed`)
- Génération automatique de labels numériques à partir des noms de fichiers
- Prétraitement d’image : découpe, binarisation, nettoyage, centrage et redimensionnement au format MNIST
- Format de données compatible avec des modèles OCR et IA

---

## 🚀 Exécution (scripts à lancer)

* Générer les chemins d’images scannées :
   `python src/prep/generate_parsed_paths.py`

* Prétraiter les images extraites :
   `python src/prep/preprocessing.py`

* Générer les chemins et labels des images traitées :
   `python src/prep/generate_custom_paths.py`
   `python src/prep/generate_custom_label.py`

* Générer les chemins pour MNIST :
   `python src/prep/generate_mnist_paths.py`

---

## 🧩 Prérequis Python

- Python 3.8 ou plus
- Dépendances listées dans le fichier `requirements.txt`

Installation recommandée :
```bash
pip install -r requirements.txt
```
