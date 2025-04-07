import os
import argparse

def collect_subimg_paths(root_folder, output_file):
    # Créer le dossier de sortie si nécessaire
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Dossier créé : {output_dir}")

    # Ouvrir le fichier et écrire les chemins
    with open(output_file, 'w') as f:
        for parent in os.listdir(root_folder):
            parent_path = os.path.join(root_folder, parent)
            subimg_path = os.path.join(parent_path, "subimg")

            if os.path.isdir(subimg_path):
                for file in os.listdir(subimg_path):
                    if file.lower().endswith(".png"):
                        full_path = os.path.join(subimg_path, file)
                        f.write(f"{full_path}\n")
    print(f"✅ Chemins collectés dans : {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Génère les chemins des images dans tous les subimg/")
    parser.add_argument('--input', required=True, help="Dossier racine (ex: parsed-200dpi/)")
    parser.add_argument('--output', required=True, help="Fichier de sortie (ex: subimg_paths.txt)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    collect_subimg_paths(args.input, args.output)


