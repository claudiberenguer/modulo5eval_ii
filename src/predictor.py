import pandas as pd
import joblib
import src.config as config


def cargar_modelo_entrenado(ruta_modelo=config.BEST_MODEL_PATH):
    """
    Carga el modelo de Machine Learning
    """
    try:
        modelo = joblib.load(ruta_modelo)
        print(f"Modelo cargado con éxito desde: {ruta_modelo}")
        return modelo
    except FileNotFoundError:
        print(f"Error: No se ha encontrado ningún modelo en la ruta {ruta_modelo}.")
        print("Entrena y guarda el modelo primero.")
        return None

def predecir_cancelacion(modelo, nuevos_datos):
    """
    Devuelve un DataFrame limpio con la predicción y la probabilidad.
    """
    # si cancela (1) o no (0)
    predicciones = modelo.predict(nuevos_datos)
    
    probabilidades = modelo.predict_proba(nuevos_datos)[:, 1]
    
    resultados = pd.DataFrame({
        'Prediccion_Numerica': predicciones,
        'Probabilidad_Cancelacion': probabilidades
    })
    
    resultados['Alerta_Negocio'] = resultados['Prediccion_Numerica'].apply(
        lambda x: 'Peligro de Cancelación' if x == 1 else 'Reserva Segura'
    )
    
    resultados['Probabilidad_Cancelacion'] = (resultados['Probabilidad_Cancelacion'] * 100).round(2).astype(str) + '%'
    
    return resultados