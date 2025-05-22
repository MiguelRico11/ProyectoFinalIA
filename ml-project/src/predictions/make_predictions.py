import joblib
import pandas as pd

def load_model(model_path):
    """Carga el modelo entrenado desde el archivo especificado."""
    model = joblib.load(model_path)
    return model

def make_prediction(model, input_data):
    """Realiza una predicción utilizando el modelo cargado y los datos de entrada proporcionados."""
    prediction = model.predict(input_data)
    return prediction

def main():
    # Ruta del modelo entrenado
    model_path = 'src/decision_tree/decision_tree_model.pkl'
    
    # Cargar el modelo
    model = load_model(model_path)
    
    # Datos de entrada para la predicción (ejemplo)
    input_data = pd.DataFrame([{
        'feature1': 10,
        'feature2': 20,
        'feature3': 30,
        'feature4': 40
    }])
    
    # Realizar la predicción
    prediction = make_prediction(model, input_data)
    
    print(f'Predicción: {prediction[0]}')

if __name__ == "__main__":
    main()