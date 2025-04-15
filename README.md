# Projet BE_25 - HEKZAM OCR

Ce projet a pour but de traiter des images scannées issues de copies d'étudiants, de détecter les chiffres manuscrits (notamment les numéros d'étudiant), et de les reconnaître automatiquement à l’aide de moteurs OCR ou de modèles IA (Tesseract, EasyOCR, Keras...).

## License
- Code: Apache-2.0
- Everything else, in particular documentation and measurements: CC-BY-SA-4.0

**Contributeurs :**
- Paulin LAURENT
- Yennayer BENAMEUR
- Nourhane MALLEK
- Omar KARAM




- **src** :
   	- **prep** : sources pour le preprocessing des fichiers/des images
- **resources** : dossiers d'images et fichiers de paths/labels
	- **parsed-200dpi** : données saisies et scannées de BE 2024
	- **likeMnist** : images issues des données manuscrites remplies lors de BE 2024
	- **mnist** : images issues du jeu de donnée MNIST de Yann Le Cun
- **results** : fichiers de comparaison entre chaque algorithme

## Structure du projet
OCR-EVAL-25/ 
├── resources/ 
│	 ├── parsed-200dpi/ # Données scannées brutes
│	 ├── mnist/ # Extraits du dataset MNIST original 
│	 ├── custom/ # Images prétraitées prêtes pour l'entraînement
│	 ├── paths_*.txt # Fichiers contenant les chemins vers les images 
│	 ├── mnist_label.txt # Labels associés aux images mnist 
│	 └── custom_label.txt # Labels associés aux images custom 
├── src/ 
│	 └── prep/
│		 ├── generate_mnist_paths.py 
│		 ├── generate_custom_paths.py 
│		 ├── generate_custom_label.py 
│		 ├── generate_parsed_paths.py 
│		 └── preprocessing.py 
├── results/ # Résultats d’évaluation des différents modèles 
└── README.md




---

## ⚙️ Fonctionnalités principales

- 🗂 Génération de chemins d’accès pour les datasets (`mnist`, `custom`, `parsed`)
- 🏷 Génération de labels pour les chiffres détectés
- 🧼 Prétraitement d’image : découpe, binarisation, nettoyage, redimensionnement au format MNIST
- 🧠 Prêt pour l’entraînement de modèles OCR classiques et IA

---

## 🚀 Exécution

1. **Extraire les chemins des images :**
   ```bash
   python src/prep/generate_parsed_paths.py
