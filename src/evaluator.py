# Para interpretar los resultados

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)


def calcular_metricas(modelo, X_test, y_test, nombre_modelo):
    """
    Calcula las medidas de rendimiento clave enfocadas en la clase 1 (cancelaciones).
    """
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]
    
    resultado = pd.DataFrame(data=[
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, pos_label=1),
        recall_score(y_test, y_pred, pos_label=1),
        f1_score(y_test, y_pred, pos_label=1),
        roc_auc_score(y_test, y_pred_proba)
    ],
    index=['Accuracy', 'Precision (Clase 1)', 'Recall (Clase 1)', 'F1-score (Clase 1)', 'AUC (Clase 1)'],
    columns=[nombre_modelo])
    
    # Formato porcentaje para que sea más legible
    resultado = (resultado * 100).round(2).astype(str) + '%'                          
    return resultado 

def evaluar_modelo(modelo, X_train, X_test, y_train, y_test, nombre_modelo):
    """
    Genera un reporte completo: métricas de train/test, Matriz de Confusión y Curva ROC.
    """
    sns.set_theme(font_scale=1.2)
    
    # 1. Reportes en texto plano por consola
    y_pred_train = modelo.predict(X_train)
    print(f"\n\t  Reporte de clasificación (Entrenamiento) - {nombre_modelo}")
    print("-" * 65)
    print(classification_report(y_train, y_pred_train))
    
    y_pred_test = modelo.predict(X_test)
    print(f"\n\t  Reporte de clasificación (Test) - {nombre_modelo}")
    print("-" * 65)
    print(classification_report(y_test, y_pred_test))
    
    # 2. Configuración del lienzo para los 3 gráficos
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), dpi=100, gridspec_kw={'width_ratios': [2, 2, 1]})
    
    # Definimos un mapa de color corporativo (tonos azules)
    royalblue = LinearSegmentedColormap.from_list('royalblue', [(0, (1,1,1)), (1, (0.25,0.41,0.88))])
    royalblue_r = royalblue.reversed()
    
    # 3. Gráfico 1: Matriz de confusión
    ConfusionMatrixDisplay.from_estimator(modelo, X_test, y_test, colorbar=False, cmap=royalblue_r, ax=ax1)
    ax1.set_title('Matriz de Confusión (Test)')                                     
    ax1.grid(False)
    
    # 4. Gráfico 2: Curva ROC
    RocCurveDisplay.from_estimator(modelo, X_test, y_test, ax=ax2)
    ax2.set_xlabel('Tasa de Falsos Positivos')
    ax2.set_ylabel('Tasa de Verdaderos Positivos')
    ax2.set_title('Curva ROC (Test)')
    
    # 5. Gráfico 3: Tabla de métricas
    resultado_metricas = calcular_metricas(modelo, X_test, y_test, nombre_modelo)
    table = ax3.table(cellText=resultado_metricas.values, colLabels=resultado_metricas.columns, rowLabels=resultado_metricas.index, loc='center')
    table.scale(0.8, 2)
    table.set_fontsize(12)
    ax3.axis('tight')
    ax3.axis('off')
    
    # Pintamos la cabecera de la tabla de azul
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_color('royalblue')
            cell.get_text().set_color('white')
            
    plt.tight_layout()
    plt.show()