import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os

# ------------------------ 1. CARGAR BASE DE DATOS ------------------------
print("📥 Descargando dataset...")
path = kagglehub.dataset_download("harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows")
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
df = pd.read_csv(os.path.join(path, csv_files[0]))
print(f"✅ Dataset cargado: {csv_files[0]} con {df.shape[0]} filas y {df.shape[1]} columnas\n")

# ------------------------ 2. LIMPIEZA BÁSICA ------------------------
df['Gross'] = df['Gross'].str.replace(',', '').astype(float)
df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(float)
df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')

# ------------------------ 3. ANÁLISIS DE CADA COLUMNA ------------------------
print("🔍 Análisis detallado por columna:\n")
for col in df.columns:
    print(f"📌 Columna: {col}")
    print(f"  - Tipo de dato: {df[col].dtype}")
    print(f"  - Valores únicos: {df[col].nunique()}")
    print(f"  - Nulos: {df[col].isnull().sum()} / {len(df)}")
    if df[col].dtype in ['float64', 'int64']:
        print(f"  - Media: {df[col].mean():.2f}")
        print(f"  - Mediana: {df[col].median():.2f}")
        print(f"  - Varianza: {df[col].var():.2f}")
        print(f"  - Mínimo: {df[col].min()} | Máximo: {df[col].max()}")
        print(f"  - Valores en 0: {(df[col] == 0).sum()}")
    print("-" * 60)

# ------------------------ 4. VARIANZA Y VISUALIZACIÓN ------------------------
varianzas = df.select_dtypes(include='number').var()
print("\n📊 Varianza por columna numérica:\n", varianzas, "\n")

plt.figure(figsize=(10, 5))
sns.barplot(x=varianzas.index, y=varianzas.values)
plt.title("Varianza de columnas numéricas")
plt.ylabel("Varianza")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------ 5. VALORES NULOS Y CEROS ------------------------
nulos = df.isnull().sum()
ceros = (df.select_dtypes(include='number') == 0).sum()

print("🧩 Valores nulos por columna:")
print(nulos[nulos > 0], "\n")

print("🪫 Valores en 0 por columna numérica:")
print(ceros[ceros > 0], "\n")

# Visualizar nulos con heatmap
plt.figure(figsize=(12, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='coolwarm')
plt.title("Mapa de calor de valores nulos")
plt.tight_layout()
plt.show()

# ------------------------ 6. DECISIONES INTELIGENTES ------------------------
print("🤖 Decisiones sugeridas por columna:\n")
for col in df.columns:
    nulls = df[col].isnull().sum()
    total = len(df[col])
    if nulls > 0:
        porc = nulls / total
        if porc > 0.5:
            print(f"⚠️  '{col}': +50% de nulos → considerar eliminar.")
        elif df[col].dtype in ['float64', 'int64']:
            print(f"🔧 '{col}': rellenar nulos con media o mediana.")
        else:
            print(f"🔧 '{col}': rellenar nulos con el valor más común (modo).")
    if df[col].dtype in ['float64', 'int64'] and (df[col] == 0).sum() > 0:
        print(f"❗ '{col}': tiene ceros → revisar si representan datos válidos o faltantes.")