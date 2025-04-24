import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import kagglehub
import os
import joblib

# ------------------------ 1. CARGA DE DATOS ------------------------
print("📥 Descargando dataset desde Kaggle...")
path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")

csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
df = pd.read_csv(os.path.join(path, csv_files[0]))

print(f"✅ Dataset cargado: {csv_files[0]} con {df.shape[0]} filas y {df.shape[1]} columnas\n")

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

# ------------------------ 3. ENTRENAMIENTO CON GRID MANUAL ------------------------
results = []
n_estimators_values = [50, 100]
max_depth_values = [5, 10]
min_samples_split_values = [2, 5]

for n_estimators in n_estimators_values:
    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            model.fit(X_train, y_train)
            pred_val = model.predict(X_val)
            mse = mean_squared_error(y_val, pred_val)
            r2 = r2_score(y_val, pred_val)

            results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'MSE': mse,
                'R2': r2
            })

results_df = pd.DataFrame(results)
print(results_df)

# ------------------------ 4. MEJOR MODELO ------------------------
best_params = results_df.loc[results_df['R2'].idxmax()]
print("Mejores hiperparámetros:\n", best_params)

X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

best_model = RandomForestRegressor(
    n_estimators=int(best_params['n_estimators']),
    max_depth=int(best_params['max_depth']) if not pd.isnull(best_params['max_depth']) else None,
    min_samples_split=int(best_params['min_samples_split']),
    random_state=42
)
best_model.fit(X_train_val, y_train_val)

# ------------------------ 5. EVALUACIÓN FINAL ------------------------
test_pred = best_model.predict(X_test)
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

pred_gross = best_model.predict(nuevo)
print(f"Predicted Gross: {pred_gross[0]}")

# Guardar y cargar modelo
joblib.dump(best_model, 'random_forest_gross_model.pkl')
print("Modelo guardado correctamente.")
loaded_model = joblib.load('random_forest_gross_model.pkl')
print(f"Predicción con modelo cargado: {loaded_model.predict(nuevo)[0]}")

# ------------------------ 7. GRAFICOS ------------------------

# 7.1. Real vs Predicho
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=test_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Valor Real de Gross')
plt.ylabel('Valor Predicho de Gross')
plt.title('🎯 Valor Real vs Valor Predicho (Test Set)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7.2. Importancia de características
importances = best_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('📊 Importancia de Características del Modelo')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.tight_layout()
plt.show()

# 7.3. Distribución de Gross
plt.figure(figsize=(8, 6))
sns.histplot(y, bins=30, kde=True)
plt.title('📈 Distribución del Ingreso Bruto (Gross)')
plt.xlabel('Gross ($)')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.show()