import os
import re

input_folder = "resources/parsed-200dpi"
output_file = "resources/paths_parsed.txt"

# Fonction pour trier les fichiers d’un dossier par leur index numérique
def extract_digit_index(filename):
    match = re.search(r'digit\.(\d+)\.(\d+)\.png', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (9999, 9999)

# Liste finale des chemins triés
collected_paths = []

# On parcourt les dossiers 0/, 1/, etc. dans l'ordre
for folder in sorted(os.listdir(input_folder), key=lambda x: int(x) if x.isdigit() else x):
    subimg_path = os.path.join(input_folder, folder, "subimg")
    if not os.path.isdir(subimg_path):
        continue

    files = [
        f for f in os.listdir(subimg_path)
        if f.endswith(".png") and "digit" in f
    ]
    # Tri des fichiers dans CE dossier
    sorted_files = sorted(files, key=extract_digit_index)

    # Ajout des chemins complets dans la liste finale
    for file in sorted_files:
        full_path = os.path.join(subimg_path, file)
        collected_paths.append(full_path)

# Sauvegarde dans le fichier de sortie
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    for path in collected_paths:
        f.write(path + "\n")

print(f"✅ {len(collected_paths)} chemins triés dossier par dossier dans : {output_file}")
