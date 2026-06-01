import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils.validation import check_random_state

import src.config as config

def loader(dataset_csv_filename=config.RAW_DATA_PATH, OHE=True):
    '''
    dataset_csv_filename: ruta al csv con los datos
    OHE: si es true, los campos categóricos se condifican con OneHotEncoder. 
         si false los campos de texto se codifican a categoría y después a numérico usando sus códigos
    '''
    check_random_state(config.RANDOM_STATE)  # Asegura que el estado aleatorio es válido

    dset = pd.read_csv(dataset_csv_filename)

    # Eliminación de variables con fugas de datos (Data Leakage)
    columns_to_drop = [
        'reservation_status',
        'reservation_status_date',
        'arrival_date_year',
        'assigned_room_type'
    ]
    dset.drop(columns_to_drop, axis=1, inplace=True)

    # Limpieza de anomalías y outliers
    dset.loc[dset['adr'] < 0, 'adr'] = dset['adr'].median()
    dset = dset[dset['adults'] != 0]
    dset = dset[dset['children'] != 10]
    dset = dset[dset['babies'] != 10]

    # Feature Engineering
    top_countries = dset['country'].value_counts().nlargest(10).index
    dset['country'] = dset['country'].where(dset['country'].isin(top_countries), 'Other')

    # Mantener la temporalidad
    month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    dset['arrival_date_month'] = dset['arrival_date_month'].map(month_map)

    # Simplificar Agente y Compañía a variables binarias
    dset['has_company'] = dset['company'].notnull().astype(int)
    dset['has_agent'] = dset['agent'].notnull().astype(int)
    dset.drop(['company', 'agent'], axis=1, inplace=True)
    
    # Relleno de los valores faltantes restantes
    dset['children'] = dset['children'].fillna(dset['children'].median())

    # Separar la target y features
    X = dset.drop(['is_canceled'], axis=1)
    y = dset['is_canceled']

    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    # Aplicación del ColumnTransformer y Escalado
    list_one_hot_cols = []
    if OHE:
        list_one_hot_cols = [
            'hotel', 'meal', 'country', 'market_segment', 
            'distribution_channel', 'reserved_room_type', 
            'deposit_type', 'customer_type'
        ]
    else:
        text_fields = [c for c in X_train.columns if X_train[c].dtype==object]
        for c in text_fields:
            X_train[c] = X_train[c].astype('category').cat.codes
            X_test[c] = X_test[c].astype('category').cat.codes
    
    list_numeric_cols = [col for col in X_train.columns if col not in list_one_hot_cols]

    columns_preprocessor = ColumnTransformer(transformers=[
        ('one_hot', OneHotEncoder(drop='first', sparse_output=False), list_one_hot_cols),
        #('ordinal', OrdinalEncoder(), list_ordinal_cols),
        ('numeric', MinMaxScaler(), list_numeric_cols)
    ])

    X_train_processed = columns_preprocessor.fit_transform(X_train)
    X_test_processed = columns_preprocessor.transform(X_test)

    nombres_columnas = columns_preprocessor.get_feature_names_out()

    nombres_columnas = [nombre.replace('one_hot__', '').replace('numeric__', '') for nombre in nombres_columnas]

    X_train = pd.DataFrame(X_train_processed, columns=nombres_columnas, index=X_train.index)
    X_test = pd.DataFrame(X_test_processed, columns=nombres_columnas, index=X_test.index)

    return (X_train, X_test, y_train, y_test)


