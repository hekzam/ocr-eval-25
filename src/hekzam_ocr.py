import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='detect numbers from images')
    parser.add_argument('algorithm', metavar = 'algorithm', help= 'choose the algorithm to use',
                        choices=['knn', 'svm', 'rf', 'lr', 'everything'])

    args = parser.parse_args()
    test_file = pd.read_csv("resources/test_data.csv")
    train_file = pd.read_csv("resources/train_data.csv")

    x_train,y_train=train_file.iloc[:,1:].values,train_file.iloc[:,0].values
    x_test,y_test=test_file.iloc[:,1:].values,test_file.iloc[:,0].values

    #TODO faire retourner les algo dans rslt.append(['nom_algo', return_algo])
    rslt=[]
    match args.algorithm :
        case 'knn':
            print()
            #TODO appel de la fonction knn
            
        case 'svm':
            print()
            #TODO appel de la fonction svm

        case 'rf':
            print()
            #TODO appel de la fonction rf
            
        case 'lr':
            print()
            #TODO appel de la fonction lr

        case 'everything':
            print()
            #TODO appel des fonctions
        
    for i in rslt:
        print(i)
        #TODO faire la gestion des données de retour 

if __name__ == "__main__":
    main()