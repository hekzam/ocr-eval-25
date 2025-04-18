import os
import re

custom_folder = "resources/custom"
output_txt = "resources/custom_label.txt"

# Récupération et tri des fichiers PNG
files = sorted([
    f for f in os.listdir(custom_folder)
    if f.endswith(".png") and "digit" in f
])

# Extraction du label (chiffre manuscrit) à partir du nom
labels = []
for file in files:
    match = re.search(r'digit\.(\d+)\.\d+\.png', file)
    if match:
        label = int(match.group(1))
        labels.append(label)
    else:
        print(f"⚠️ Nom inattendu : {file}")
        labels.append(-1)  # Pour repérer les cas anormaux

# Écriture dans le fichier
with open(output_txt, "w") as f:
    for label in labels:
        f.write(f"{label}\n")

print(f"✅ Fichier de labels généré : {output_txt}")
print(f"🔢 Total : {len(labels)} labels")
