import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def apply_kmeans(data, n_clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_data)

    return kmeans.labels_, kmeans.inertia_

def plot_clusters(data, labels, n_clusters):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=labels, palette='viridis', s=100)
    plt.title(f'Clusters Visualisation (n_clusters={n_clusters})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Clusters')
    plt.show()

def cluster_analysis(data, max_clusters=10):
    inertia = []
    for n in range(1, max_clusters + 1):
        labels, cluster_inertia = apply_kmeans(data, n)
        inertia.append(cluster_inertia)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Cargar los datos desde el archivo
    data = pd.read_csv("/Users/carlosmejia/Documents/Universidad/Septimo semestre/IA/Proyecto 2/ProyectoFinalIA/ml-project/data/processed/processed_dataset.csv")

    # Seleccionar columnas numéricas relevantes para clustering
    numeric_columns = ["IMDB_Rating", "Meta_score", "No_of_Votes", "Gross"]
    data = data[numeric_columns].dropna()

    # Realizar análisis de clustering
    print("Realizando análisis de clustering...")
    cluster_analysis(data, max_clusters=10)

    # Aplicar KMeans con un número óptimo de clusters (por ejemplo, 3)
    print("Aplicando KMeans con 3 clusters...")
    labels, _ = apply_kmeans(data, n_clusters=3)

    # Agregar etiquetas de cluster al DataFrame
    data['Cluster'] = labels
    cluster_summary = data.groupby('Cluster').mean()
    print(cluster_summary)

    # Visualizar los clusters
    print("Visualizando los clusters...")
    plot_clusters(data, labels, n_clusters=3)