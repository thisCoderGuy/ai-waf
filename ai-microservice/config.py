# --- Model and Preprocessor Loading ---


BASE_MODEL_DIR = '/app/trained-models/'

# Define the specific filenames for the model and preprocessor
MODEL_FILENAME = 'cnn_malicious_traffic_model_20250703_100424.joblib'
PREPROCESSOR_FILENAME = 'cnn_malicious_traffic_preprocessor_20250703_100424.joblib'

# Construct the full paths by joining the base directory and filenames
import os

MODEL_PATH = os.path.join(BASE_MODEL_DIR, MODEL_FILENAME)
PREPROCESSOR_PATH = os.path.join(BASE_MODEL_DIR, PREPROCESSOR_FILENAME)
