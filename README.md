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


## Structure du projet

- **src** :
   	- **prep** : sources pour le preprocessing des fichiers/des images
- **resources** : dossiers d'images et fichiers de paths/labels
	- **parsed-200dpi** : données saisies et scannées de BE 2024
	- **likeMnist** : images issues des données manuscrites remplies lors de BE 2024
	- **mnist** : images issues du jeu de donnée MNIST de Yann Le Cun
- **results** : fichiers de comparaison entre chaque algorithme


