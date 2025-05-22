import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

def load_data(filename):
    """Carga el conjunto de datos desde la carpeta data/processed."""
    raw_data_path = os.path.join("data", "processed", filename)
    return pd.read_csv(raw_data_path)

def analyze_missing_values(df):
    """Analiza y visualiza los valores nulos en el DataFrame."""
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_values.index, y=missing_values.values)
    plt.title('Valores Nulos por Columna')
    plt.xlabel('Columnas')
    plt.ylabel('Número de Valores Nulos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_data_distribution(df, column):
    """Visualiza la distribución de una columna específica del DataFrame."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f'Distribución de {column}')
    plt.xlabel(column)
    plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()

def analyze_variance(df):
    """Analiza la varianza de las columnas numéricas del DataFrame."""
    variance = df.var()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=variance.index, y=variance.values)
    plt.title('Varianza de las Características')
    plt.xlabel('Características')
    plt.ylabel('Varianza')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def clean_data(df):
    """Realiza limpieza básica de datos, eliminando filas con valores nulos."""
    return df.dropna()

if __name__ == "__main__":
    # Cargar los datos desde la carpeta data/raw
    df = load_data("processed_dataset.csv")  # Cambia "processed_dataset.csv" por el nombre de tu archivo en data/raw

    # Analizar valores nulos
    print("Analizando valores nulos...")
    analyze_missing_values(df)

    # Visualizar la distribución de una columna específica
    print("Visualizando la distribución de la columna 'IMDB_Rating'...")
    visualize_data_distribution(df, "Meta_score")  # Cambia "IMDB_Rating" por otra columna si lo deseas

    # Analizar varianza
    print("Analizando varianza de las características...")
    analyze_variance(df)

    # Limpieza de datos
    print("Limpiando datos...")
    df_cleaned = clean_data(df)
    print("Datos limpiados:")
    print(df_cleaned.head())