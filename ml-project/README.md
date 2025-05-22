# Machine Learning Project

Este proyecto de aprendizaje automático tiene como objetivo analizar un conjunto de datos y construir modelos predictivos utilizando algoritmos de clustering y árboles de decisión. A continuación se presenta una descripción general de la estructura del proyecto y los pasos realizados.

## Estructura del Proyecto

```
ml-project
├── data
│   ├── raw
│   │   └── dataset.csv
│   └── processed
│       └── processed_dataset.csv
├── notebooks
│   └── training_notebook.ipynb
├── src
│   ├── eda
│   │   └── exploratory_data_analysis.py
│   ├── clustering
│   │   └── clustering_algorithms.py
│   ├── decision_tree
│   │   └── decision_tree_model.py
│   ├── predictions
│   │   └── make_predictions.py
│   └── utils
│       └── data_preprocessing.py
├── requirements.txt
└── README.md
```

## Descripción de Archivos

- **data/raw/dataset.csv**: Contiene el conjunto de datos original utilizado para el análisis y modelado.
- **data/processed/processed_dataset.csv**: Contiene el conjunto de datos procesado, limpio y transformado para el entrenamiento del modelo.
- **notebooks/training_notebook.ipynb**: Cuaderno de Jupyter donde se realiza el entrenamiento del modelo, incluyendo la selección de algoritmos y evaluación de métricas.
- **src/eda/exploratory_data_analysis.py**: Funciones para realizar el análisis exploratorio de datos (EDA), generando gráficos y analizando valores nulos.
- **src/clustering/clustering_algorithms.py**: Implementación de algoritmos de clustering y análisis de los clusters obtenidos.
- **src/decision_tree/decision_tree_model.py**: Implementación de un modelo de árbol de decisión, incluyendo el entrenamiento y análisis de la importancia de las variables.
- **src/predictions/make_predictions.py**: Funciones para realizar predicciones utilizando el modelo entrenado.
- **src/utils/data_preprocessing.py**: Funciones utilitarias para el preprocesamiento de datos, como la imputación de valores nulos.
- **requirements.txt**: Lista de dependencias necesarias para el proyecto.

## Instalación

Para instalar las dependencias del proyecto, ejecute el siguiente comando:

```
pip install -r requirements.txt
```

## Uso

1. Realice el análisis exploratorio de datos utilizando el archivo `exploratory_data_analysis.py`.
2. Procese los datos utilizando las funciones en `data_preprocessing.py`.
3. Aplique algoritmos de clustering en `clustering_algorithms.py`.
4. Entrene el modelo de árbol de decisión en el cuaderno `training_notebook.ipynb`.
5. Realice predicciones utilizando el archivo `make_predictions.py`.

## Conclusiones

Este proyecto proporciona un marco para el análisis de datos y la construcción de modelos predictivos utilizando técnicas de aprendizaje automático. Se espera que los resultados obtenidos ayuden a comprender mejor el conjunto de datos y a realizar predicciones precisas.