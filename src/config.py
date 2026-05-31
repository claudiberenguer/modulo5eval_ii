# El fichero config.py sirve para guardar todas las variables globales, rutas de carpetas y parámetros fijos en un solo lugar.

from pathlib import Path

# Ruta raíz del proyecto
# __file__ es este script (config.py). parent.parent nos lleva a la carpeta principal
ROOT_DIR = Path(__file__).resolve().parent.parent

# Ruta de datos
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_PATH = DATA_DIR / 'raw' / 'dataset_practica_final.csv'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Rutas de los modelos
MODELS_DIR = ROOT_DIR / 'models'
MODELS_TEST_DIR = MODELS_DIR / 'tests'
BEST_MODEL_PATH = MODELS_DIR / 'best_model.pkl'
DT_MODEL_PATH = MODELS_TEST_DIR / 'tree.pkl'

# Rutas de las salidas visuales
OUTPUTS_DIR = ROOT_DIR / 'outputs'

# Parámetros globales
RANDOM_STATE = 42
TEST_SIZE = 0.2