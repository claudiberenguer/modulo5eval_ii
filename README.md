# Predicción de Cancelaciones de Reservas Hoteleras 🏨

Este proyecto es la resolución de la práctica final del módulo de Machine Learning y Deep Learning del Máster en Inteligencia Artificial de PontIA Tech. Su objetivo es desarrollar un sistema automatizado capaz de predecir si un cliente cancelará su reserva de hotel, permitiendo al negocio tomar medidas proactivas para evitar pérdidas económicas.

---

## 1. Autores y Definición de Roles 👥

Este proyecto ha sido desarrollado en pareja, dividiendo las responsabilidades para simular un entorno de trabajo real:

* **Manuel Yerbes:** Encargado del diseño de la arquitectura de software modular (fábrica de datos, evaluación, predicción), desarrollo de los modelos de Regresión Logística y Árbol de Decisión y XGBoost, y de la implementación del bonus técnico (API REST con FastAPI).
* **Claudi Berenguer Sabaté:** Encargado de la experimentación con Random Forest, Red Neuronal Multicapa, ajustes de hiperparámetros  y validación de las métricas finales.
* **Trabajo Conjunto:** Análisis Exploratorio de Datos (EDA), selección de la métrica principal de negocio y redacción de esta documentación.

---

## 2. Descripción y Justificación del Problema 🎯

Las cancelaciones de reservas suponen un problema financiero crítico para el sector hotelero. Una habitación vacía por una cancelación de última hora representa una pérdida del 100% de los ingresos previstos para esa noche. 

**Justificación del enfoque:**
Hemos modelado esto como un problema de clasificación binaria (Clase 0: Mantiene reserva / Clase 1: Cancela reserva). Para el negocio, un falso negativo (el modelo dice que viene, pero cancela) es mucho más costoso que un falso positivo (el modelo dice que cancelará, pero finalmente viene). Por lo tanto, nuestra optimización matemática se ha centrado en maximizar el **Recall de la clase 1**, asegurando que atrapamos la mayor cantidad posible de cancelaciones antes de que ocurran, incluso si sacrificamos un pequeño porcentaje de *Accuracy* global.

---

## 3. Análisis Exploratorio de Datos (EDA) 📊

Durante la fase de exploración (detallada en `notebooks/finales/eda_final.ipynb`), identificamos y tratamos diversos retos en el dataset proporcionado:
* **Data Leakage:** Se eliminaron columnas temporales y de estado que revelarían la respuesta antes de tiempo.
* **Anomalías:** Se imputaron valores negativos en las tarifas (ADR) por la mediana y se descartaron registros ilógicos (ej. reservas con 0 adultos o 10 bebés).
* **Ingeniería de Características:** Se agruparon países con baja frecuencia, se transformaron variables como agente/compañía a formato binario y se aplicó *One-Hot Encoding* a las variables categóricas, estandarizando los valores numéricos mediante *MinMaxScaler*.

---

## 4. Diseño del Sistema (Arquitectura) ⚙️

El proyecto se ha construido siguiendo principios de ingeniería de software, separando la experimentación del código de producción.

**Estructura Principal:**
* `src/config.py`: Gestor centralizado de rutas y variables globales.
* `src/data_loader.py`: Pipeline automatizado de carga y preprocesamiento.
* `src/model_trainer.py`: Motor genérico de entrenamiento mediante `GridSearchCV`.
* `src/evaluator.py`: Generador automático de reportes, Matrices de Confusión y curvas ROC.
* `src/predictor.py`: Interfaz para cargar modelos empaquetados (`.pkl`) y generar inferencias.

**Bonus Técnico (API REST):**
Hemos implementado un servidor con **FastAPI** (`main.py`) que expone el modelo final mediante endpoints `/predict` y `/evaluate`, permitiendo la integración de este sistema con cualquier aplicación web o software de recepción hotelera.

---

## 5. Instrucciones de Ejecución 🚀

Para reproducir este proyecto en un entorno local, sigue estos pasos:

**1. Clonar el repositorio y configurar el entorno:**
```bash
git clone https://github.com/claudiberenguer/modulo5eval_ii[https://github.com/claudiberenguer/modulo5eval_ii]
cd modulo5eval_ii
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Ejecutar la comparativa de modelos:**
Abre Jupyter Notebook o tu editor de preferencia y ejecuta de principio a fin el archivo notebooks/finales/comparativa_modelos.ipynb. Esto entrenará los modelos, generará los gráficos y guardará los archivos .pkl en la carpeta models/tests/.

**3. Levantar la API en producción:**
Asegúrate de estar en la raíz del proyecto y ejecuta:

```bash
uvicorn main:app --reload
```

Abre el navegador en http://127.0.0.1:8000/docs para interactuar con la interfaz Swagger y probar los endpoints.

## 6. Resultados y Elección Final 🏆
Tras automatizar el entrenamiento y ajustar los hiperparámetros (incluyendo class_weight='balanced'), obtuvimos los siguientes resultados en el conjunto de test:


| Modelo | Accuracy | Precision (C1) | Recall (C1) | F1-score (C1) | AUC-ROC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Regresión Logística Optimizada | 80.16% | 71.56% | 77.16% | 74.25% | 88.76% |
| Árbol de Decisión Optimizado | 83.36% | 74.69% | 83.36% | 78.79% | 91.20% |
| XGBoost | 86.72% | 80.02% | 85.54% | 82.69% | 94.57% |
| Random Forest | [85]% | [76]% | [82]% | [79]% | [92]% |85
| Red Neuronal | [85]% | [78]% | [83]% | [80]% | [93]% |

*Nota: Pendiente de añadir las métricas finales de Random Forest y Red Neuronal de la fase de experimentación.*

**Elección Final:** Seleccionamos **XGBoost** como el modelo principal de producción. Como se puede observar en la tabla, domina absolutamente en todas las métricas en comparación con los modelos base. Destaca especialmente su **Recall del 85.54%** (crucial para cazar las cancelaciones sorpresa y proteger financieramente al hotel) y un altísimo **AUC-ROC del 94.57%**, lo que demuestra una capacidad sobresaliente y muy superior al resto para distinguir patrones complejos entre clientes que mantendrán su reserva y los que cancelarán. Todo esto, logrando además el mejor equilibrio con las falsas alarmas (Precision del 80.02%).

## 7. Reflexión Crítica y Limitaciones 🧠

- Limitaciones del Dataset: El modelo asume que el comportamiento futuro de los clientes será idéntico al del pasado. Factores externos cruciales que provocan cancelaciones (meteorología adversa, cancelación de vuelos o crisis económicas) no están recogidos en los datos actuales, lo que limita su capacidad predictiva en escenarios excepcionales.

- Mejoras Futuras: 1. Si tuviéramos mayor capacidad computacional, ampliaríamos el espacio de búsqueda del GridSearchCV para la Red Neuronal Multicapa.
2. Incorporaríamos variables contextuales (ej. festivos en el país de origen).
3. Para una puesta en producción real completa (CI/CD), contenerizaríamos la API con Docker y la desplegaríamos en un servicio Cloud.
