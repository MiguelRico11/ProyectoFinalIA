import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import kagglehub
import os
import numpy as np

# ------------------------ 1. CARGA DE DATOS ------------------------
print("Descargando dataset desde Kaggle...")
path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
df = pd.read_csv(os.path.join(path, csv_files[0]))
print(f"Dataset cargado: {csv_files[0]} con {df.shape[0]} filas y {df.shape[1]} columnas\n")

# Limpieza
df['Gross'] = df['Gross'].str.replace(',', '').astype(float)
df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(float)
df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
df = df[['Runtime', 'Released_Year', 'IMDB_Rating', 'Meta_score', 'No_of_Votes', 'Gross']].dropna()

# ------------------------ 2. ESCALADO DE DATOS ------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop(columns=['Gross']))

# ------------------------ 3. K-MEANS CLUSTERING ------------------------
k = 5  # Número de clústeres
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# ------------------------ 4. ANÁLISIS DE CLÚSTERES ------------------------
cluster_summary = df.groupby('Cluster')['Gross'].agg(['mean', 'count']).reset_index()
print("Ingreso bruto promedio por clúster:\n", cluster_summary)

# ------------------------ 5. VISUALIZACIÓN ------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
df['PCA1'] = pca_data[:, 0]
df['PCA2'] = pca_data[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='tab10')
plt.title("🎬 Películas agrupadas por similitud (PCA + KMeans)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.tight_layout()
plt.show()

# ------------------------ 6. PREDICCIÓN PARA NUEVA PELÍCULA ------------------------
nueva_peli = pd.DataFrame([{
    'Runtime': 120,
    'Released_Year': 2020,
    'IMDB_Rating': 8.0,
    'Meta_score': 75,
    'No_of_Votes': 500000
}])
nueva_scaled = scaler.transform(nueva_peli)
clust_pred = kmeans.predict(nueva_scaled)[0]
gross_aprox = cluster_summary.loc[cluster_summary['Cluster'] == clust_pred, 'mean'].values[0]

print(f"\n🎯 La nueva película cae en el clúster {clust_pred} con un ingreso bruto promedio estimado de: ${gross_aprox:,.2f}")