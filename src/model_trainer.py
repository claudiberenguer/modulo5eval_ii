# sirve para guardar funciones genéricas de entrenamiento
from sklearn.model_selection import StratifiedKFold, GridSearchCV

def optimizar_hiperparametros_modelo(modelo_base, cuadricula_parametros, X_train, y_train, metrica_evaluacion='f1', num_divisiones=5):
    """
    Entrena un modelo evaluando múltiples combinaciones de hiperparámetros.
    
    Parámetros:
    - modelo_base: El algoritmo a entrenar (ej. DecisionTreeClassifier()).
    - cuadricula_parametros: Diccionario con los parámetros a probar.
    - X_train, y_train: Datos de entrenamiento.
    - metrica_evaluacion: Métrica para decidir qué modelo gana (por defecto 'f1').
    - num_divisiones: Número de particiones para la validación cruzada.
    
    Retorna:
    - El modelo ya optimizado y la lista de sus mejores hiperparámetros.
    """
    # Crear la estrategia de validación cruzada garantizando la proporción de cancelaciones
    cross_validation = StratifiedKFold(n_splits=num_divisiones, shuffle=True, random_state=0)

    # Configurar el buscador automático
    buscador_gridsearch = GridSearchCV(
        estimator=modelo_base, 
        param_grid=cuadricula_parametros, 
        cv=cross_validation, 
        scoring=metrica_evaluacion, 
        n_jobs=-1
    )

    buscador_gridsearch.fit(X_train, y_train)

    # resultados ganadores
    mejores_hiperparametros = buscador_gridsearch.best_params_

    return buscador_gridsearch.best_estimator_, mejores_hiperparametros






