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

# Construct the full paths by joining the base directory and filenames
MODEL_PATH = os.path.join(BASE_MODEL_DIR, MODEL_FILENAME)
PREPROCESSOR_PATH = os.path.join(BASE_MODEL_DIR, PREPROCESSOR_FILENAME)

# Possible Log LEvels: DEBUG, INFO, WARNING, ERROR, and CRITICAL.
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',                       # Choose DEBUG or INFO to see appropriate messages in the console
            'formatter': 'standard',
        },
    },

    'loggers': {
        'evaluation_logger': {
            'level': 'INFO',                            
            'handlers': ['console'],
            'propagate': False
        },
        'global_logger': {
            'level': 'DEBUG',                            
            'handlers': ['console'],
            'propagate': False
        }
    }
}

#########################################################
# --- Directory Setup (created if they don't exist) ---
#########################################################
# Directories required for the project structure
REQUIRED_DIRS = [
    os.path.join('training', 'training-data', 'raw'),
    os.path.join('training', 'training-data', 'cleaned'),    
    os.path.join('training', 'model-logs'),     
    os.path.join('ai-microservice', 'trained-models')
]