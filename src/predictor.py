import pandas as pd
import joblib
import src.config as config


def cargar_modelo_entrenado(ruta_modelo=config.BEST_MODEL_PATH):
    """
    Carga un modelo de Machine Learning previamente entrenado y guardado.
    """
    try:
        # joblib abre el archivo .pkl y lo devuelve como un modelo de Python funcional
        modelo = joblib.load(ruta_modelo)
        print(f"Modelo cargado con éxito desde: {ruta_modelo}")
        return modelo
    except FileNotFoundError:
        print(f"Error: No se ha encontrado ningún modelo en la ruta {ruta_modelo}.")
        print("Asegúrate de haber entrenado y guardado el modelo primero.")
        return None

def predecir_cancelacion(modelo, nuevos_datos):
    """
    Recibe un modelo entrenado y un DataFrame con nuevas reservas.
    Devuelve un DataFrame limpio con la predicción y la probabilidad.
    """
    # 1. El modelo decide si cancela (1) o no (0)
    predicciones = modelo.predict(nuevos_datos)
    
    # 2. El modelo calcula el % de seguridad de esa decisión
    probabilidades = modelo.predict_proba(nuevos_datos)[:, 1]
    
    # 3. Montamos una tabla bonita pensada para el departamento de negocio del hotel
    resultados = pd.DataFrame({
        'Prediccion_Numerica': predicciones,
        'Probabilidad_Cancelacion': probabilidades
    })
    
    # 4. Traducimos los 0s y 1s a texto para que sea 100% intuitivo
    resultados['Alerta_Negocio'] = resultados['Prediccion_Numerica'].apply(
        lambda x: 'Peligro de Cancelación' if x == 1 else 'Reserva Segura'
    )
    
    # Lo ponemos en formato porcentaje (ej: 85.5%)
    resultados['Probabilidad_Cancelacion'] = (resultados['Probabilidad_Cancelacion'] * 100).round(2).astype(str) + '%'
    
    return resultados