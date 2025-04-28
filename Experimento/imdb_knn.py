import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import kagglehub
import os
import joblib

# ------------------------ 1. CARGA DE DATOS ------------------------
print("Descargando dataset desde Kaggle...")
path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")

csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
df = pd.read_csv(os.path.join(path, csv_files[0]))

print(f"Dataset cargado: {csv_files[0]} con {df.shape[0]} filas y {df.shape[1]} columnas\n")

# Limpieza básica
df['Gross'] = df['Gross'].str.replace(',', '').astype(float)
df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(float)
df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')

# Elimina filas con columnas clave nulas
df = df[['Runtime', 'Released_Year', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross']].dropna()

# ------------------------ 2. SEPARACIÓN DE DATOS ------------------------
X = df.drop(columns=['Gross'])
y = df['Gross']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ------------------------ 3. ENTRENAMIENTO CON KNN ------------------------
results = []
k_values = [3, 5, 7, 9, 11]

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred_val = knn.predict(X_val)
    mse = mean_squared_error(y_val, pred_val)
    r2 = r2_score(y_val, pred_val)

    results.append({
        'k': k,
        'MSE': mse,
        'R2': r2
    })

results_df = pd.DataFrame(results)
print(results_df)

# ------------------------ 4. MEJOR MODELO ------------------------
best_params = results_df.loc[results_df['R2'].idxmax()]
print("Mejor valor de k:\n", best_params)

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

best_knn = KNeighborsRegressor(n_neighbors=int(best_params['k']))
best_knn.fit(X_train_val, y_train_val)

# ------------------------ 5. EVALUACIÓN FINAL ------------------------
test_pred = best_knn.predict(X_test)
print("Test MSE:", mean_squared_error(y_test, test_pred))
print("Test R2:", r2_score(y_test, test_pred))

# ------------------------ 6. PREDICCIÓN NUEVO DATO ------------------------
nuevo = pd.DataFrame([{
    'Runtime': 120,
    'Released_Year': 2020,
    'IMDB_Rating': 8.0,
    'Meta_score': 75,
    'No_of_Votes': 500000
}])

pred_gross = best_knn.predict(nuevo)
print(f"Predicted Gross: {pred_gross[0]}")

# Guardar y cargar modelo
joblib.dump(best_knn, 'knn_gross_model.pkl')
print("Modelo guardado correctamente.")
loaded_model = joblib.load('knn_gross_model.pkl')
print(f"Predicción con modelo cargado: {loaded_model.predict(nuevo)[0]}")
