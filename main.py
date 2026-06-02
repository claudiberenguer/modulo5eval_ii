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
    Recibe datos en JSON y devuelve si van a cancelar o si no.
    """
    try:
        # Cambiar config.DT_MODEL_PATH para el árbol de decición, config.RL_MODEL_PATH para la regresión logística o config.XGB_MODEL_PATH para XGBoost
        modelo = cargar_modelo_entrenado(config.XGB_MODEL_PATH)
        if modelo is None:
            raise HTTPException(status_code=500, detail="Modelo no encontrado en el servidor.")

        df_nuevos_datos = pd.DataFrame(reservas.datos)

        resultados = predecir_cancelacion(modelo, df_nuevos_datos)

        return {"predicciones": resultados.to_dict(orient="records")}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/evaluate")
def evaluate():
    """
    Evalúa el modelo actual en producción.
    """
    try:
        modelo = cargar_modelo_entrenado(config.XGB_MODEL_PATH)
        
        _, X_test, _, y_test = loader(OHE=True)
        
        metricas = calcular_metricas(modelo, X_test, y_test, 'Regresión Logística API')
        
        return {"metricas": metricas.to_dict()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
