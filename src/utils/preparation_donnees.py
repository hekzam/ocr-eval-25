import numpy as np
import pandas as pd
from PIL import Image

def extract_path_and_label(txtPath, txtLabel):
    count =0
    lTrain= len(train_list)
    lTest= len(test_list)
    with open(txtPath) as f:
        count = sum(1 for _ in f)
    with open(txtPath) as f:
        for line_nb, line in enumerate(f, start=0):
            line = line[:-1]
            if count/6<line_nb:
                train_list.append([line])
            else:
                test_list.append([line])
    with open(txtLabel) as f:
        for line_nb, line in enumerate(f, start=0):
            line = line[:-1]
            if count/6<line_nb:
                train_list[lTrain].append(int(line))
                lTrain+=1
            else:
                test_list[lTest].append(int(line))
                lTest+=1

def traitement_image(image_path, label):
    img = Image.open(image_path).convert('L') 
    img = img.resize((28, 28))  #pas nécessaire mais ajouter une sécurité
    img_array = np.array(img)
    pixels = img_array.flatten()
    return [label] + pixels.tolist()

if __name__ == "__main__":
    #rend les données maniable
    custom_path = "resources/paths_custom.txt"
    custom_label = "resources/custom_label.txt" 
    mnist_path = "resources/paths_mnist.txt"
    mnist_label = "resources/mnist_label.txt" 

    csv_test_path = "resources/test_data.csv"
    csv_train_path = "resources/train_data.csv"

    train_list = []
    test_list = []

    #TODO: remettre cette ligne de code quand probleme de syncro des txt de custom est réglé
    #extract_path_and_label(custom_path,custom_label)
    extract_path_and_label(mnist_path,mnist_label)
    
    trainData = []
    for i in train_list:
        trainData.append(traitement_image(i[0], i[1]))

    testData = []
    for i in test_list:
        testData.append(traitement_image(i[0], i[1]))
    # sauvegarde en csv 
    
    columns = ['label'] + [f'pixel{i}' for i in range(28*28)]

    df_train = pd.DataFrame(trainData, columns=columns)
    df_train.to_csv(csv_train_path, index=False)

    df_test = pd.DataFrame(testData, columns=columns)
    df_test.to_csv(csv_test_path, index=False)

    print("sauvegarde en csv effectuée")