from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

import src.config as config
from src.data_loader import loader
from src.evaluator import calcular_metricas
from src.predictor import cargar_modelo_entrenado, predecir_cancelacion

app = FastAPI(
    title="Hotel Cancellation API",
    description="API REST para predecir cancelaciones de reservas hoteleras.",
    version="1.0.0"
)


class DatosReserva(BaseModel):
    datos: list[dict]

@app.get("/")
def home():
    """Comprobación de salud de la API."""
    return {"mensaje": "¡Bienvenido a la API de Predicción de Cancelaciones del Hotel!"}

@app.post("/predict")
def predict(reservas: DatosReserva):
    """
    Recibe datos de nuevas reservas en formato JSON y devuelve si van a cancelar.
    """
    try:
        # Cambiar config.DT_MODEL_PATH para el árbol de decición o config.RL_MODEL_PATH para la regresión logística
        modelo = cargar_modelo_entrenado(config.RL_MODEL_PATH)
        if modelo is None:
            raise HTTPException(status_code=500, detail="Modelo no encontrado en el servidor.")

        # De JSON a DataFrame
        df_nuevos_datos = pd.DataFrame(reservas.datos)

        resultados = predecir_cancelacion(modelo, df_nuevos_datos)

        # Resultado en JSON
        return {"predicciones": resultados.to_dict(orient="records")}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/evaluate")
def evaluate():
    """
    Carga los datos de testeo y evalúa el modelo actual en producción.
    """
    try:
        # Cargar el modelo
        modelo = cargar_modelo_entrenado(config.RL_MODEL_PATH)
        
        _, X_test, _, y_test = loader(OHE=True)
        
        metricas = calcular_metricas(modelo, X_test, y_test, 'Regresión Logística API')
        
        return {"metricas": metricas.to_dict()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
"""
No se pone el endpoint /train porque eso bloquearía la API durante el proceso de entrenamiento, que puede ser largo. En un entorno real, el entrenamiento se haría por separado y se actualizaría el modelo en producción sin afectar a los usuarios.
"""

# Encender la api con: uvicorn main:app --reload