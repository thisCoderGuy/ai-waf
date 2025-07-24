import os
# --- Model and Preprocessor Loading ---
BASE_MODEL_DIR = '/app/trained-models/'

# store the filename of the most recently trained model and its preprocessor. This file will reside within the ai-microservice directory.
LATEST_MODEL_INFO_PATH = os.path.join(BASE_MODEL_DIR, 'latest_model_info.txt')

# --- Existing Model and Preprocessor Loading (for explicit provision) ---
# If you need a way to explicitly load a *specific* model
# rather than always the latest, e.g., for testing or rollback.

# Used as a fallback or explicit override.
# Define the specific filenames for the model and preprocessor
MODEL_FILENAME = 'cnn_malicious_traffic_model_20250703_100424.joblib'
PREPROCESSOR_FILENAME = 'cnn_malicious_traffic_preprocessor_20250703_100424.joblib'
TRAINING_TYPE = 'deep' # 'deep' or 'traditional'
MODEL_TYPE = 'cnn'

# Construct the full paths by joining the base directory and filenames
MODEL_PATH = os.path.join(BASE_MODEL_DIR, MODEL_FILENAME)
PREPROCESSOR_PATH = os.path.join(BASE_MODEL_DIR, PREPROCESSOR_FILENAME)

