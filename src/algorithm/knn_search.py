import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# charger les données complètes
df = pd.read_parquet("resources/data_utilisees/train_data.parquet")
x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# échantillon pour le tuning
x_small, _, y_small, _ = train_test_split(x, y, train_size=5000, stratify=y, random_state=42)

# recherche du meilleur k entre 1 et 9
param_grid = {'n_neighbors': list(range(1, 10))}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(x_small, y_small)

best_k = grid.best_params_['n_neighbors']
print(f"meilleur k trouvé: {best_k}")

# test de performance selon différentes tailles d'entraînement
subset_sizes = [10000, 20000, 30000, 40000, 50000, 60000]
print("\n evaluation des tailles d'entrainement :")

x_test = pd.read_parquet("resources/data_utilisees/test_data.parquet").iloc[:, 1:].values
y_test = pd.read_parquet("resources/data_utilisees/test_data.parquet").iloc[:, 0].values

for size in subset_sizes:
    x_sub = x[:size]
    y_sub = y[:size]

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(x_sub, y_sub)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    print(f"- {size:5d} exemples → Accuracy : {acc:.4f}")