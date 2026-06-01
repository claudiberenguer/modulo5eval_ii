from sklearn.model_selection import StratifiedKFold, GridSearchCV

def optimizar_hiperparametros_modelo(modelo_base, cuadricula_parametros, X_train, y_train, metrica_evaluacion='f1', num_divisiones=5):
    """
    Entrena un modelo evaluando múltiples combinaciones de hiperparámetros.
    
    Devuelve el modelo optimizado y la lista de sus mejores hiperparámetros.
    """
    cross_validation = StratifiedKFold(n_splits=num_divisiones, shuffle=True, random_state=0)

    buscador_gridsearch = GridSearchCV(
        estimator=modelo_base, 
        param_grid=cuadricula_parametros, 
        cv=cross_validation, 
        scoring=metrica_evaluacion, 
        n_jobs=-1
    )

    buscador_gridsearch.fit(X_train, y_train)

    mejores_hiperparametros = buscador_gridsearch.best_params_

    return buscador_gridsearch.best_estimator_, mejores_hiperparametros






