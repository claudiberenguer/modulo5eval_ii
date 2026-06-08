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

El análisis de los datos se ha llevado a cabo en paralelo, poniendo en común lo observado por cada uno en los notebooks `notebooks/exploracion/eda_inicial_claudi.ipynb` y `evaluacion_manuel`, para decidir de forma conjunta el tratamiento final. Este último también incorpora entrenamiento y evaluación de LogisticRegression, DeisionTree y XGBoost.
Las conclusiones son las siguientes y se encuentran como un DataFrame en `columns_info` en `data_loader.py`, que contiene la función `loader()` que procesa los datos de entrada de acuerdo a lo concluido en la exploración inicial devolviendo `X_tran`, `X_test`, `y_train`, `y_test` (se hace un tratameinto de los datos unificado para todos los modelos: filtro de features, imputación, procesado de categorías, one-hot encoder en features categóricas no ordinales y MinMaxScaler en el resto):
| Feature                        | Type | KD   | Comment                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------ | ---- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| hotel                          | cat  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| is_canceled                    | cat  | keep | target, dataset ligeramente desbalanceado (63%, 37%)                                                                                                                                                                                                                                                                                                                 |
| lead_time                      | num  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| arrival_date_year              | ord  | drop | el año en que se hizo la reserva no tiene valor para predecir a futuro. En todo caso se usa para ver si hay correlación entre cancelado/no_cancelado y el año; si existiera, el dataset no sería coherente y debería fraccionarse en partes homogéneas usando la más reciente. En este caso no se observa esa correlación, por lo que se mantiene el dataset entero. |
| arrival_date_month             | ord  | drop | redundante con arrival_date_week_number                                                                                                                                                                                                                                                                                                                              |
| arrival_date_week_number       | ord  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| arrival_date_day_of_month      | ord  | drop | redundante con arrival_date_week_number                                                                                                                                                                                                                                                                                                                              |
| stays_in_weekend_nights        | num  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| stays_in_week_nights           | num  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| adults                         | num  | keep | drop rows with 0                                                                                                                                                                                                                                                                                                                                                     |
| children                       | num  | drop | está muy desbalanceada y en entrenamiento RFC se ha visto que está entre las variables menos significativas; se quita obteniendo ligera mejora en recall                                                                                                                                                                                                             |
| babies                         | num  | drop | drop >8. está muy desbalanceada y en entrenamiento RFC se ha visto que está entre las variables menos significativas; se quita obteniendo ligera mejora en recall                                                                                                                                                                                                    |
| meal                           | cat  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| country                        | cat  | keep | muchas categorías con baja frecuencia; se agrupan en "other"                                                                                                                                                                                                                                                                                                         |
| market_segment                 | cat  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| distribution_channel           | cat  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| is_repeated_guest              | cat  | keep | muy desbalanceada                                                                                                                                                                                                                                                                                                                                                    |
| previous_cancellations         | num  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| previous_bookings_not_canceled | num  | keep | conjuntamente con previous_cancellations da un ratio de cancelación pasado por cliente                                                                                                                                                                                                                                                                               |
| reserved_room_type             | cat  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| assigned_room_type             | cat  | keep | la reserva puede ser cancelada in-situ si la habitación asignada es distinta de la reservada y no satisface expectativas                                                                                                                                                                                                                                             |
| booking_changes                | num  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| deposit_type                   | cat  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| agent                          | cat  | keep | NaN interpretado como cliente sin agente. Muchas clases de baja frecuencia; se agrupan las menos frecuentes en "sin agente"                                                                                                                                                                                                                                          |
| company                        | cat  | keep | NaN interpretado como no viaje por empresa; se convierte en binaria (empresa vs no empresa)                                                                                                                                                                                                                                                                          |
| days_in_waiting_list           | num  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| customer_type                  | cat  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| adr                            | num  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| required_car_parking_spaces    | num  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| total_of_special_requests      | num  | keep |                                                                                                                                                                                                                                                                                                                                                                      |
| reservation_status             | cat  | drop | irrelevante; se conoce después de la target (is_canceled). Introduce data leakage                                                                                                                                                                                                                                                                                    |
| reservation_status_date        | cat  | drop | irrelevante; derivada de reservation_status                                                                                                                                                                                                                                                                           
Las princpales evaluaciones han sido las siguientes:
- Cheqeo de dependencia del target con la feature arrival_date_year: se ve que lo hay y por lo tanto el dataset es coherente y se puede usar en su totalidad
- chequeo del balanceo del dataset: 67% '0' (is_not_canceled) / 37% '1' (is_canceled)): ligreramiente desbalanceado, esto se va a tener en cuenta tanto en al hacer el train_test_split (mantener un frecuencia parecida tante en train como en test) como a la hora de entrenar los modelos (dando mayor peso a las instancias de la clase minoritaria)
- dado que la mayoría de las variables son categóricas, se hace un plot de barras de las frecuencias de cada feature por cada clase (is_canceled / is_not_canceled), para ver a cuales la clasificación es sensible, así como para ver su frecuencia y despersión
- se ven qué clases tiene valores faltantes. En agent y company tienen un significado, que es 'no se ha usado agente' y 'el cliente no viaja a cargo de una compañía'
- se lleva a cabo una codificación numérica de de los campos categóricos de texto en base a transformar la columna a categoría y tomar su código, pero no se usa en el tratamiento final en tanto que introduce ordinalidad artifical. Se decide usar one hot encoder para las variables catgóricas. En aquellas con un número elevado de elementos de baja frecuencia, previamente se van a agrupar estas últimas como 'other' (casos country y agent). Otras de este mismo tipo se han convertido en binarias (caso company)
- Se lleva a cabo una detección de outlayers en las features de tipo contínuo (numérico) en base a estadística Z. se ve que el resultado no es coherente y se descarta
- A raíz de lo anterior se lleva a cabo lo mismo mediante clustering con DBSCAN, en tanto que el algoritmo implementa el concepto de 'no alcanzable', no requiere dar un número cluseter de entrada y ajusta epsilon automáticamente. Requiere de escalado, para evitar que features con rangos mayores dominen al resto. Se prueba con escalado por MInMaxScaler y StandardScaler. Para dar un epsilon inicial se mide la distancia media en un subconjunto del dataset esogido aleatoriamente y se usa la mitad. Con MinMaxScaler se obtienen valores razonables, on StandarScaler no. Salen pocos outlayers y no se van a quitar. Aun así en base al resultado de este ejercicio se ha decidio usar MinMaxScaler.
- En el entrenamiento con RandomForestClassifier se saca la importancia de las variables. En base a ello finalmente se ha decidido sacar algunas muy desbalanceadas (baby y children)
- Imputación de valores faltantes en campos numéricos por la mediana (la media teine el peligro de estar desvirutada por algunos outlayers); también en valores ilógicos (como número de adultos a 0 o valores negativos en el precio de la habituación)
---

## 4. Diseño del Sistema (Arquitectura) ⚙️

El proyecto se ha construido siguiendo principios de ingeniería de software, separando la experimentación del código de producción.

**Estructura Principal:**

* `src/config.py`: Gestor centralizado de rutas y variables globales.
* `src/data_loader.py`: Pipeline automatizado de carga y preprocesamiento.
* `src/model_trainer.py`: Motor genérico de entrenamiento mediante `GridSearchCV`.
* `src/evaluator.py`: Generador automático de reportes, Matrices de Confusión y curvas ROC.
* `src/predictor.py`: Interfaz para cargar modelos empaquetados (`.pkl`) y generar inferencias.

Dado que el modelo de red neuronal sigue un proceso de entrenamiento distinto y también reporta los datos de entrenamiento de forma distinta, lo anterior se ha llevado a cabo a parte, conjuntamente con RandomForestClassifier, en en noteook `finales/newnb_claudi (loader definition and NN & RFC training and evaluation)`. 


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
| Random Forest | 85% | 76% | 82% | 79% | 92% |85
| Red Neuronal | 85% | 78% | 83% | 80% | 93% |


**Elección Final:** Seleccionamos **XGBoost** como el modelo principal de producción. Como se puede observar en la tabla, domina absolutamente en todas las métricas en comparación con los modelos base. Destaca especialmente su **Recall del 85.54%** (crucial para cazar las cancelaciones sorpresa y proteger financieramente al hotel) y un altísimo **AUC-ROC del 94.57%**, lo que demuestra una capacidad sobresaliente y muy superior al resto para distinguir patrones complejos entre clientes que mantendrán su reserva y los que cancelarán. Todo esto, logrando además el mejor equilibrio con las falsas alarmas (Precision del 80.02%).

## 7. Reflexión Crítica y Limitaciones 🧠

- Limitaciones del Dataset: El modelo asume que el comportamiento futuro de los clientes será idéntico al del pasado. Factores externos cruciales que provocan cancelaciones (meteorología adversa, cancelación de vuelos o crisis económicas) no están recogidos en los datos actuales, lo que limita su capacidad predictiva en escenarios excepcionales.

- Mejoras Futuras: 1. Si tuviéramos mayor capacidad computacional, ampliaríamos el espacio de búsqueda del GridSearchCV para la Red Neuronal Multicapa.
2. Incorporaríamos variables contextuales (ej. festivos en el país de origen).
3. Para una puesta en producción real completa (CI/CD), contenerizaríamos la API con Docker y la desplegaríamos en un servicio Cloud.
