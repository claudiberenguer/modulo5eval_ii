# Función para yo pedirle los datos y esta función, me los devuelve ya limpios y listos para usar en el modelo.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import config as config

from sklearn.utils.validation import check_random_state
check_random_state(42)

## columns_info resume la información de cada columna del dataset, su tipo, si se mantiene o se elimina y un comentario sobre su tratamiento. Es una herramienta útil para documentar las decisiones tomadas durante la limpieza y preprocesamiento de los datos.
## es un dataframe con las columnas 'type' (tipo de dato, cat, ord, num), 'kd' (keep o drop) y 'comment' (comentarios sobre el tratamiento de la columna). Se ha creado a mano a partir del análisis exploratorio de los datos y se utiliza como referencia para el proceso de limpieza y preprocesamiento en la función loader.
## se puede importar desde este módulo para tener toda la información sobre las columnas en un solo lugar, lo que facilita la comprensión y el mantenimiento del código. Además, se puede actualizar fácilmente si se decide cambiar el tratamiento de alguna columna en el futuro.
columns_info = pd.DataFrame({
'hotel':['cat', 'keep', ''],
'is_canceled':['cat', 'keep', 'target, 63% 0, 37% 1 (algo desbalanceado pero no demasiado)'],
'lead_time':['num', 'keep', ''],
'arrival_date_year':['ord', 'drop', 'el año de cancelación no tiene valor para predecir el futuro. En todo caso se usa para ver si la distribución del target es homogénea por años; se ha comrobado que sí,por lo que se puede tomar la totalidad del dataset, toods los años'],
'arrival_date_month':['ord', 'drop', 'redundante con arrival_date_week_number'],
'arrival_date_week_number':['ord', 'keep', ''],
'arrival_date_day_of_month':['ord', 'keep', 'redundante con arrival_dat_week_number'],
'stays_in_weekend_nights':['num', 'keep', ''],
'stays_in_week_nights':['num', 'keep', ''],
'adults':['num', 'drop','drop rows with 0'],
'children':['num', 'drop', ''],
'babies':['num', 'drop', 'drop rows with >8'],
'meal':['cat', 'keep',''],
'country':['cat', 'keep', 'muchas categorías, la mayoría con baja frecuencia. Se van a agrupar las menos frecuentes en una categoría "Other", así como los nan'],
'market_segment':['cat', 'keep', ''],
'distribution_channel':['cat', 'keep', ''],
'is_repeated_guest':['cat', 'keep', 'fuertemente desbalanceada, solo el 3% de los clientes son recurrentes. Se mantiene porque puede ser un indicador de cancelación, pero se podría haber eliminado por su baja frecuencia'],
'previous_cancellations':['num','keep',''],
'previous_bookings_not_canceled':['num', 'keep','conjuntamente con previous_cancellations da el ratio de cancelaciones previas, que puede ser un indicador de cancelación futura'],
'reserved_room_type':['cat', 'keep', ''],
'assigned_room_type':['cat', 'keep', 'la reserva pude ser cancelada a última hora si la habitación asignada no es la reservada (reserved_room_type)'],
'booking_changes':['num','keep',''],
'deposit_type':['cat','keep',''],
'agent':['cat', 'keep', 'nan se interpreta como que el cliente no ha sido gestionado por un agente. Con muchas categorías., muchas de ellas con baja frecuencia. Se prueba a convertir el cambo a binario (por agente/sin agente) y a tomas todos los agentes con mayor frecuencia que el caso "sin agente", finalmente optando por la segunda'],
'company':['cat','keep','nan se interpreta que el cliente no es de negocios, esto es, que viene por empresa. muy desbalanceada y con muchas categorías de baja frecuencia. se transforma en binaria (con empresa/sin empresa)'],
'days_in_waiting_list':['num', 'keep', ''],
'customer_type':['cat', 'keep', ''],
'adr':['num', 'keep', ''],
'required_car_parking_spaces':['num','keep',''],
'total_of_special_requests':['num', 'keep',''],
'reservation_status':['cat', 'drop','irrelevante porqué se da una vez el target ya ha sid; provocaría el efecto llamado data leakage, es decir, que el modelo aprenda a predecir a partir de información que no estaría disponible en el momento de la predicción'], 
'reservation_status_date':['cat','drop', 'irrelevante, propiedad de la anterior, que es irrelevante']}).T
columns_info.columns = ['type', 'kd', 'comment']


def loader(columns_info, OHE:bool) -> (pd.core.frame.DataFrame, pd.core.frame.DataFrame, pd.core.series.Series, pd.core.series.Series):
    '''
    dataset_csv_filename: ruta al csv con los datos
    OHE: si es true, los campos categóricos se condifican con OneHotEncoder. 
         si false los campos de texto se codifican a categoría y después a numérico usando sus códigos
    '''

    # carga del csv en un dataframe
    dset = pd.read_csv(config.RAW_DATA_PATH)

    # se aletatoriza la posición de las filas del dataset
    dset = dset.sample(frac=1).reset_index(drop=True)
    
    # Eliminación de features inconvenientes o irrelevantes según columns_info
    columns_to_drop = columns_info[columns_info.kd=='drop'].index
    dset.drop(columns_to_drop, axis= 1, inplace=True)

    # --- LIMPIEZA DE ANOMALÍAS ---
    
    # Reemplazar el valor negativo por la mediana
    if columns_info.loc['adr','kd'] == 'keep':
        dset.loc[dset['adr'] < 0, 'adr'] = dset['adr'].median()

    # Eliminar las filas con 0 adultos
    if columns_info.loc['adults','kd'] == 'keep':
        dset = dset[dset['adults'] != 0]

    # Eliminar las filas con 10 niños y mas de 8 bebes (en base a value counts y númeoro de adultos de esos filas, que son casos claramente extremos)
    if columns_info.loc['children','kd'] == 'keep':
        dset = dset[dset['children'] != 10]
    if columns_info.loc['babies','kd'] == 'keep':
        dset = dset[dset['babies'] < 8]

    # Agrupar países con poca representación
    if columns_info.loc['country','kd'] == 'keep':
        top_countries = dset['country'].value_counts().nlargest(10).index
        dset['country'] = dset['country'].where(dset['country'].isin(top_countries), 'Other')

    # Simplificar 'comany' a variable binaria (con empresa/sin empresa)'
    if columns_info.loc['company','kd'] == 'keep':
        dset['has_company'] = dset['company'].notnull().astype(int)
    
    # agupar agentes con poca representación. El caso 'sin agente' está a na. 
    if columns_info.loc['agent','kd'] == 'keep':
        # las reservas que no han usado agente están a na. Se sustituye por 0, que es un código no usado
        dset['agent'] = dset['agent'].fillna(0)
        # Agrupar agentes con poca representación. El caso 'sin agente' está en el ranking 11. Se le asigna ell código 40, que no està entre los 10 de mayor frecuéncia
        top_agents = dset['agent'].value_counts().nlargest(11).index
        dset['agent'] = dset['agent'].where(dset['agent'].isin(top_agents), 40)
    
    # relleno de los valores faltantes en 'children' con la mediana, ya que es una variable numérica y no se puede asumir que el valor faltante signifique 0 niños. Además, la mediana es una medida de tendencia central robusta que no se ve afectada por valores extremos, lo que la hace adecuada para este caso.
    if columns_info.loc['children','kd'] == 'keep':    
        dset['children'] = dset['children'].fillna(dset['children'].median())

    # separación en target (variable objetivo) y features
    y = dset['is_canceled']
    X = dset.drop(['is_canceled'], axis=1)
   
    # OneHot encoder de las variables categóricas (no ordinales) si el parámetro OHE es True.
    # Si el parámetro OHE es False, se convierten los campos de texto a categorías y después a códigos numéricos. 
    list_one_hot_cols = []
    if OHE:
        list_one_hot_cols = [c for c in X.columns if c in columns_info[columns_info.type=='cat'].index]
    else:
        # primero se ponen en una lista los cmps de texto
        # después se convierten los campos a catgorías, se toman sus códigos numéricos (.cat.codes) y se sustitye a los string originales por ellos
        text_fields = [c for c in X.columns if X_train[c].dtype==object]
        for c in text_fields:
            X[c] = X[c].astype('category').cat.codes  

    # la lista de campos numéricos es la complementaria a la de campos a codificar con OneHotEncoder, por lo que incluye los ordinales (son de fecha en este caso)  
    list_numeric_cols = [col for col in X.columns if col not in list_one_hot_cols]

    columns_preprocessor = ColumnTransformer(transformers=[
        ('one_hot', OneHotEncoder(drop='first', sparse_output=False), list_one_hot_cols),
        #('one_hot', OneHotEncoder(drop='if_binary', sparse_output=False), list_one_hot_cols),
        #('ordinal', OrdinalEncoder(), list_ordinal_cols),
        ('numeric', MinMaxScaler(), list_numeric_cols)
    ])

    # fit_transform en el conjunto entero de features. No se separa por evitar el caso de categorías que por lo que fuese estuviesen en X_train pero no en X_test, lo que haría que OnedHotEncoder resultase en shapes distintas para train y test. 
    X_processed_arr = columns_preprocessor.fit_transform(X)

    # Restaurar a DataFrame para no perder los nombres, emmpezando por obtener los nombres de las columnas después de la transformación.
    nombres_columnas = columns_preprocessor.get_feature_names_out()
    
    # Se limpia el texto que añade scikit-learn por defecto
    nombres_columnas = [nombre.replace('one_hot__', '').replace('numeric__', '') for nombre in nombres_columnas]

    X_processed = pd.DataFrame(X_processed_arr, columns=nombres_columnas, index=X.index)

    # train_test_split se hace antes de trasnformar nada. Se incluye stratify para asegurar que la distribución del target sea homogénea en train y test. 
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y)
    
    return (X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = loader(columns_info, OHE=True)
    pass
   


