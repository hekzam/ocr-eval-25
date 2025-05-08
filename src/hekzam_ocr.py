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
    parser.add_argument('--train', help= 'specify whether to train the model',
                        action='store_true')

    args = parser.parse_args()

    file_model_knn = "resources/models/knn_model.pkl" 
    file_model_svm = "resources/models/svm_model.pkl"
    file_model_rf = "resources/models/rf_model.pkl"
    file_model_lr = "resources/models/lr_model.pkl"

    df_test = pd.read_parquet("resources/data_utilisees/test_data.parquet")
    df_train = pd.read_parquet("resources/data_utilisees/train_data.parquet")

    x_train,y_train=df_train.iloc[:,1:].values,df_train.iloc[:,0].values
    x_test,y_test=df_test.iloc[:,1:].values,df_test.iloc[:,0].values

    #TODO faire retourner les algo dans rslt.append(['nom_algo', return_algo, temps execution algo])
    #TODO detection de train fait ou pas fait, return en fonction "merci de faire l'entrainement avant de lancer le model x"
    rslt=[]
    match args.algorithm :
        case 'knn':
            if args.train:
                start = time.perf_counter()
                knn.train(x_train, y_train, 3)
                stop = time.perf_counter()
                print("temps d'entrainement: "+ str(stop-start))
            rslt.append(calcul_algo("knn", knn, x_test))
            
        case 'svm':
            if args.train:
                start = time.perf_counter()
                svm.train(x_train, y_train)
                stop = time.perf_counter()
                print("temps d'entrainement: "+ str(stop-start))
            rslt.append(calcul_algo("svm", svm, x_test))

        case 'rf':
            if args.train:
                start = time.perf_counter()
                rf.train(x_train, y_train)
                stop = time.perf_counter()
                print("temps d'entrainement: "+ str(stop-start))
            rslt.append(calcul_algo("random forest", rf, x_test))

        case 'lr':
            if args.train:
                start = time.perf_counter()
                lr.train(x_train, y_train)
                stop = time.perf_counter()
                print("temps d'entrainement: "+ str(stop-start))
            rslt.append(calcul_algo("logistic regression", lr, x_test))

        case 'everything':
            print()
            #TODO appel des fonctions
        
    for i in rslt:
        print(i)
        #TODO faire la gestion des données de retour 

def calcul_algo(nomAlgo, modeleAlgo, x_test):
    rslt = [nomAlgo]
    start = time.perf_counter()
    rslt.append(modeleAlgo.predict(x_test))
    stop = time.perf_counter()
    rslt.append(stop-start)
    print("calcul "+nomAlgo+" terminé")
    return rslt
    

if __name__ == "__main__":
    main()