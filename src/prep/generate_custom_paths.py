import os
import re

input_folder = "resources/custom"
output_file = "resources/paths_custom.txt"

# Récupération et tri des fichiers PNG
files = sorted([
    f for f in os.listdir(input_folder)
    if f.endswith(".png") and "digit" in f
])

# Fonction qui extrait les deux index : X et Y dans raw-0-digit.X.Y.png
def extract_double_index(filename):
    match = re.search(r'digit\.(\d+)\.(\d+)\.png', filename)
    if match:
        x = int(match.group(1))  # groupe
        y = int(match.group(2))  # index
        return (x, y)
    return (9999, 9999)  # cas par défaut si non matché

# Récupération et tri des fichiers PNG
file_list = sorted([
    f for f in os.listdir(input_folder)
    if f.endswith(".png") and "digit" in f
])

# Écriture dans le fichier
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w") as f:
    for file in file_list:
        full_path = os.path.join(input_folder, file)
        f.write(full_path + "\n")

print(f"✅ {len(file_list)} chemins triés par double index écrits dans : {output_file}")
