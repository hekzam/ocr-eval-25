import os

mnist_folder = "resources/mnist"
output_file = "resources/paths_mnist.txt"

with open(output_file, "w") as f:
    for file in os.listdir(mnist_folder):
        if file.lower().endswith(".png"):
            path = os.path.join(mnist_folder, file)
            f.write(path + "\n")

print(f"✅ Chemins mnist collectés dans : {output_file}")
