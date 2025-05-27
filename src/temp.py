import pandas as pd
import visualisation.intervalle_confiance as int_conf
import visualisation.optimum_pareto as optimum

knn_csv = "results/results_algo/resultats_knn.csv"
svm_csv = "results/results_algo/resultats_svm.csv"
lr_csv = "results/results_algo/resultats_random_forest.csv"
rf_csv = "results/results_algo/resultats_logistic_regression.csv"

int_conf.construire_intervalle_confiance(pd.read_csv(knn_csv),
                                        pd.read_csv(lr_csv), pd.read_csv(rf_csv), 'results/visuels')
optimum.construire_optimum(knn_csv, lr_csv, rf_csv, 'results/visuels')