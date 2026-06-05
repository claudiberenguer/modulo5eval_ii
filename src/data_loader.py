import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import src.config as config

from sklearn.utils.validation import check_random_state
check_random_state(42)

## columns_info resume la información de cada columna del dataset, su tipo, si se mantiene o se elimina y un comentario sobre su tratamiento. Es una herramienta útil para documentar las decisiones tomadas durante la limpieza y preprocesamiento de los datos.
## es un dataframe con las columnas 'type' (tipo de dato, cat, ord, num), 'kd' (keep o drop) y 'comment' (comentarios sobre el tratamiento de la columna). Se ha creado a mano a partir del análisis exploratorio de los datos y se utiliza como referencia para el proceso de limpieza y preprocesamiento en la función loader.
## se puede importar desde este módulo para tener toda la información sobre las columnas en un solo lugar, lo que facilita la comprensión y el mantenimiento del código. Además, se puede actualizar fácilmente si se decide cambiar el tratamiento de alguna columna en el futuro.
columns_info = pd.DataFrame({
'hotel':['cat', 'keep', ''],
'is_canceled':['cat', 'keep', 'target, balanced dataset (63% 0, 37% 1, suspiciosly too much cancelations'],
'lead_time':['num', 'keep', ''],
'arrival_date_year':['ord', 'drop', 'misleading, those are years in the past, not useful for prediction, they will never come back. For the dataset to be consistent, the target should be independent ofo it'],
'arrival_date_month':['ord', 'drop', 'redundant with arrival_dat_week_number'],
'arrival_date_week_number':['ord', 'keep', ''],
'arrival_date_day_of_month':['ord', 'drop', 'redundant with arrival_dat_week_number'],
'stays_in_weekend_nights':['num', 'keep', ''],
'stays_in_week_nights':['num', 'keep', ''],
'adults':['num', 'keep','drop rows with 0'],
'children':['num', 'drop', 'está muy desbalanceada y en entrenamiento RFC se ha visto que estáa entre las variables menos significativasa, se quita, obteniendo una ligera mejora en el recall'],
'babies':['num', 'drop', 'drop >8. está muy desbalanceada y en entrenamiento RFC se ha visto que estáa entre las variables menos significativasa, se quita, obteniendo una ligera mejora en el recall'],
'meal':['cat', 'keep',''],
'country':['cat', 'keep', 'mucahs categorías, muchas de ellas con baja frecuencia. Se agrupan en "other" las características con menos frecuencia'],
'market_segment':['cat', 'keep', ''],
'distribution_channel':['cat', 'keep', ''],
'is_repeated_guest':['cat', 'keep', 'muy desbalanceada'],
'previous_cancellations':['num','keep',''],
'previous_bookings_not_canceled':['num', 'keep','conjuntamente con previous_cancellations da un ratio de cancelación pasado por cliente'],
'reserved_room_type':['cat', 'keep', ''],
'assigned_room_type':['cat', 'keep', 'la reserva puede ser cancelado in-situ si la habitación asignada es distinta de la reservada y no satisface las expectativas del cliente'],
'booking_changes':['num','keep',''],
'deposit_type':['cat','keep',''],
'agent':['cat', 'keep', 'nan se interpreta como que el clilente no ha usado agente. muchas clases, muchas de baja frecuencia. todas las clases menos frecuentes que la "sin agtente" se agrupan en una'],
'company':['cat','keep','nan se interpreta como que el cliente no viaja por empresa. fueremente desbalanceada y atomizada. se convierte en binaria (si es cliene tipo empresa o no)'],
'days_in_waiting_list':['num', 'keep', ''],
'customer_type':['cat', 'keep', ''],
'adr':['num', 'keep', ''],
'required_car_parking_spaces':['num','keep',''],
'total_of_special_requests':['num', 'keep',''],
'reservation_status':['cat', 'drop','irrelevante, se da después de haberse dado la target (is_canceled). de inmcluirse se incurriría en data leakage'], 
'reservation_status_date':['cat','drop', 'irrelevante, es una propiedad de una irrelevante (la de justo arriba)']}).T
columns_info.columns = ['type', 'kd', 'comment']
columns_info


def loader(OHE:bool) -> (pd.core.frame.DataFrame, pd.core.frame.DataFrame, pd.core.series.Series, pd.core.series.Series):
    '''
    dataset_csv_filename: ruta al csv con los datos
    OHE: si es true, los campos categóricos se condifican con OneHotEncoder. 
         si false los campos de texto se codifican a categoría y después a numérico usando sus códigos
    '''

    dset = pd.read_csv(config.RAW_DATA_PATH)

    dset = dset.sample(frac=1).reset_index(drop=True)
    
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
        text_fields = [c for c in X.columns if X[c].dtype==object]
        for c in text_fields:
            X[c] = X[c].astype('category').cat.codes  

    # la lista de campos numéricos es la complementaria a la de campos a codificar con OneHotEncoder, por lo que incluye los ordinales (son de fecha en este caso)  
    list_numeric_cols = [col for col in X.columns if col not in list_one_hot_cols]

    columns_preprocessor = ColumnTransformer(transformers=[
        ('one_hot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), list_one_hot_cols), # Aprende solo del X_train con handle_unknown
        ('numeric', MinMaxScaler(), list_numeric_cols)
    ])
    
    # train_test_split se hace antes de trasnformar nada. Se incluye stratify para asegurar que la distribución del target sea homogénea en train y test. 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    

    # fit_transform en el conjunto entero de features. No se separa por evitar el caso de categorías que por lo que fuese estuviesen en X_train pero no en X_test, lo que haría que OnedHotEncoder resultase en shapes distintas para train y test. 
    X_train_processed_arr = columns_preprocessor.fit_transform(X_train)

    X_test_processed_arr = columns_preprocessor.transform(X_test)

    # Restaurar a DataFrame para no perder los nombres, emmpezando por obtener los nombres de las columnas después de la transformación.
    nombres_columnas = columns_preprocessor.get_feature_names_out()
    
    # Se limpia el texto que añade scikit-learn por defecto
    nombres_columnas = [nombre.replace('one_hot__', '').replace('numeric__', '') for nombre in nombres_columnas]

    X_train = pd.DataFrame(X_train_processed_arr, columns=nombres_columnas, index=X_train.index)
    X_test = pd.DataFrame(X_test_processed_arr, columns=nombres_columnas, index=X_test.index)
    
    return (X_train, X_test, y_train, y_test)

   


