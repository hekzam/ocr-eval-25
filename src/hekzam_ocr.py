import argparse
import pandas as pd
import time
import algorithm.knn as knn
import algorithm.logistic_regression as lr
import algorithm.random_forest as rf
import algorithm.svm as svm
"""
commande d'exécution (algorithm_choice= knn/svm/lr/rf):
python heckzam_ocr.py algorithm_choice
si ca ne marche pas essayer celle ci (en modifiant les liens interne pour correspondre):
& path/to/python.exe path/to/hekzam_ocr.py algorithm_choice
& C:/Users/pauli/AppData/Local/Programs/Python/Python313/python.exe c:/Users/pauli/Documents/git/ocr-eval-25/src/hekzam_ocr.py rf
"""
def main():
    parser = argparse.ArgumentParser(description='detect numbers from images')
    parser.add_argument('algorithm', metavar = 'algorithm', help= 'choose the algorithm to use',
                        choices=['knn', 'svm', 'rf', 'lr', 'everything'])

    args = parser.parse_args()

    df_test = pd.read_parquet("resources/data_utilisees/test_data.parquet")
    df_train = pd.read_parquet("resources/data_utilisees/train_data.parquet")

    x_train,y_train=df_train.iloc[:,1:].values,df_train.iloc[:,0].values
    x_test,y_test=df_test.iloc[:,1:].values,df_test.iloc[:,0].values

    #TODO faire retourner les algo dans rslt.append(['nom_algo', return_algo, temps execution algo])
    #TODO faire la mesure du temps d'exécution a partir d'ici pour une méthode claire
    rslt=[]
    match args.algorithm :
        case 'knn':
            
            rslt.append(["knn"])
            i=len(rslt)-1
            start = time.perf_counter()
            rslt[i].append(knn.predict(x_test))
            stop = time.perf_counter()
            rslt[i].append(stop-start)
            print("calcul knn terminé")

            
        case 'svm':

            rslt.append(["svm"])
            i=len(rslt)-1
            start = time.perf_counter()
            rslt[i].append(svm.predict(x_test))
            stop = time.perf_counter()
            rslt[i].append(stop-start)
            print("calcul svm terminé")

        case 'rf':
            
            rslt.append(["random forest"])
            i=len(rslt)-1
            start = time.perf_counter()
            rslt[i].append(rf.predict(x_test))
            stop = time.perf_counter()
            rslt[i].append(stop-start)
            print("calcul random forest terminé")
            
        case 'lr':

            rslt.append(["logistic regression"])
            i=len(rslt)-1
            start = time.perf_counter()
            rslt[i].append(lr.predict(x_test))
            stop = time.perf_counter()
            rslt[i].append(stop-start)
            print("calcul logistic regression terminé")

        case 'everything':
            print()
            #TODO appel des fonctions
        
    for i in rslt:
        print(i)
        #TODO faire la gestion des données de retour 

if __name__ == "__main__":
    main()