import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_data(file_path):
    return pd.read_csv(file_path)

def save_data(df, file_path):
    df.to_csv(file_path, index=False)

def preprocess_data(df):
    # Imputación de valores nulos
    imputer = SimpleImputer(strategy='mean')
    df[['column1', 'column2']] = imputer.fit_transform(df[['column1', 'column2']])
    
    # Transformación de datos a formatos numéricos
    df['categorical_column'] = df['categorical_column'].astype('category').cat.codes
    
    # Escalado de características
    scaler = StandardScaler()
    df[['column1', 'column2']] = scaler.fit_transform(df[['column1', 'column2']])
    
    return df

def drop_unnecessary_columns(df, columns_to_drop):
    return df.drop(columns=columns_to_drop)