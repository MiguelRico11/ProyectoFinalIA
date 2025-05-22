import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_decision_tree_model(data_path):
    # Cargar el conjunto de datos procesado
    df = pd.read_csv(data_path)

    # Seleccionar las columnas relevantes
    selected_columns = ['Runtime', 'Released_Year', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross']
    df = df[selected_columns].dropna()  # Asegurarse de eliminar filas con valores nulos

    # Separar características y objetivo
    X = df.drop(columns=['Gross'])  # Características
    y = df['Gross']  # Columna objetivo

    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar el modelo de árbol de decisión
    model = DecisionTreeRegressor(random_state=42)

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    # Guardar el modelo entrenado
    joblib.dump(model, 'decision_tree_model.pkl')
    print("Modelo de árbol de decisión guardado correctamente.")

if __name__ == "__main__":
    train_decision_tree_model('/Users/carlosmejia/Documents/Universidad/Septimo semestre/IA/Proyecto 2/ProyectoFinalIA/ml-project/data/processed/processed_dataset.csv')