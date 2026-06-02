from pathlib import Path

# Ruta raíz del proyecto
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
RL_MODEL_PATH = MODELS_TEST_DIR / 'logistic_regression.pkl'
RFC_MODEL_PATH = MODELS_TEST_DIR / 'RandomForestClassifierModel.pkl'
NN_MODEL_PATH = MODELS_TEST_DIR / 'NeuralNetworkModel.pkl'

# Rutas de las salidas visuales
OUTPUTS_DIR = ROOT_DIR / 'outputs'

# Parámetros globales
RANDOM_STATE = 42
TEST_SIZE = 0.2