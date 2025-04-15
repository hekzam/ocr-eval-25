import os
import re

mnist_folder = "resources/mnist"
output_file = "resources/paths_mnist.txt"

def extract_number(filename):
    """Extrait le numéro depuis 'image_XXXXX.png' """
    match = re.search(r'image_(\d+)\.png', filename)
    return int(match.group(1)) if match else -1

# Liste triée par numéro
file_list = sorted(
    [f for f in os.listdir(mnist_folder) if f.lower().endswith(".png")],
    key=extract_number
)

# Sauvegarde dans le fichier .txt
with open(output_file, "w") as f:
    for file in file_list:
        full_path = os.path.join(mnist_folder, file)
        f.write(full_path + "\n")

print(f"✅ {len(file_list)} chemins triés numériquement dans : {output_file}")
