import os

# --- Model Selection and Specific Parameters ---
# Set the current model type. 
# Possible values so far: 'svm', 'random_forest', 'decision_tree', 'naive_bayes', 'cnn', 'rnn', 'lstm', 'transformer', 'llm'
MODEL_TYPE = 'naive_bayes'

# --- General Model Training Configuration ---
# Number of splits for Stratified K-Fold Cross-Validation (used by traditional models)
N_SPLITS_CROSS_VALIDATION = 5
# Test size vs training size
TEST_SIZE = 0.2
# Random state for reproducibility in data splitting and model training
RANDOM_STATE = 42

# Hyperparameter tuning:
PERFORM_TUNING = True
TUNING_METHOD = 'grid' # 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
RANDOM_SEARCH_N_ITER = 10 # Number of parameter settings that are sampled if using RandomizedSearchCV

# --- Feature Extraction Parameters (for TfidfVectorizer) ---
TFIDF_MAX_FEATURES = 5000
TFIDF_ANALYZER = 'char' # 'word' or 'char'
TFIDF_NGRAM_RANGE = (2, 4) # For character n-grams

# Parameters (when doing hyper parameter tuning, those are the base parameters being optimized) for traditional Scikit-learn models 
SKLEARN_MODEL_PARAMS = {
    'svm': {
        'kernel': 'linear',
        'probability': True,
        'C': 1.0 # Regularization parameter, example
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None, # None means nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        'random_state': RANDOM_STATE
    },
    'decision_tree': {
        'max_depth': None,
        'random_state': RANDOM_STATE
    },
    'naive_bayes': {
        # No specific parameters commonly tuned for MultinomialNB or GaussianNB, depending on data
    }
}

# Parameters for hyperparameter tuning
TUNING_PARAMS = {
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf']
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }
    # Add tuning parameters for other models here
}

# Parameters for Deep Learning Models (CNN, RNN, LSTM, Transformer)
DEEP_LEARNING_MODEL_PARAMS = {
    'cnn': {
        'input_shape': (5000, 1), # Example input shape, adjust based on feature extractor output
        'num_filters': 128,
        'kernel_size': 5,
        'pool_size': 2,
        'dense_units': 64,
        'learning_rate': 0.001,
        'epochs': 10,
        'batch_size': 32
    },
    'rnn': {
        'units': 128,
        'return_sequences': False,
        'learning_rate': 0.001,
        'epochs': 10,
        'batch_size': 32
    },
    'lstm': {
        'units': 128,
        'return_sequences': False,
        'learning_rate': 0.001,
        'epochs': 10,
        'batch_size': 32
    },
    'transformer': {
        'num_heads': 4,
        'ff_dim': 128,
        'num_blocks': 2,
        'learning_rate': 0.001,
        'epochs': 10,
        'batch_size': 32
    }
}

# Parameters for Large Language Models (LLM)
LLM_MODEL_PARAMS = {
    'llm': {
        'model_name': 'gemini-2.0-flash', # Or other LLM models
        'temperature': 0.7,
        'max_tokens': 150,
        # Any other specific LLM parameters like fine-tuning settings, API keys etc.
    }
}


# Columns to drop during initial data cleaning
COLUMNS_TO_DROP = [
    'Timestamp', 'TransactionID', 'ClientIP', 'ClientPort', 'ServerIP', 'ServerPort',
    'ResponseProtocol', 'ResponseHeaders', 'ResponseBody',
    'WAFInterrupted', 'InterruptionRuleID', 'InterruptionStatus', 'InterruptionAction',
    'MatchedRulesCount', 'MatchedRulesIDs', 'MatchedRulesMessages', 'MatchedRulesTags',
    'AIScore', 'AIVerdict' # 'TimeOfDayHour', 'TimeOfDayDayOfWeek' if they were present
]

# Specific problematic endings to filter out from raw log lines
PROBLEMATIC_ENDINGS = [
    "Failed to write CSV record: short write",
    "CSV writer error: short write"
]

# --- Configuration for Data, Model, and Log Paths ---
# Path to raw dataset files (Coraza audit log files)
LOG_FILE_PATHS = [
    os.path.join('training_data', 'raw', 'coraza-audit-benign.csv'),
    os.path.join('training_data', 'raw', 'coraza-audit-sqli.csv'),
    os.path.join('training_data', 'raw', 'coraza-audit-xss.csv')
]

# Path to save the cleaned dataset (within 'training_data' folder)
CLEANED_DATA_OUTPUT_PATH = os.path.join('training_data', 'cleaned', 'coraza-audit-cleaned.csv')

# Path to save the trained model.
MODEL_BASE_OUTPUT_DIR = os.path.join('..', 'ai-microservice')
# Prefix for the generated model and preprocessor filenames (e.g., 'svm_malicious_traffic_model_20250611_1330.joblib')
MODEL_FILENAME_PREFIX = 'malicious_traffic_model'
PREPROCESSOR_FILENAME_PREFIX = 'malicious_traffic_preprocessor'


# Path for the evaluation log file
EVALUATION_LOG_PATH = os.path.join('logs', 'evaluation_metrics.log')

# --- Directory Setup (created if they don't exist) ---
# Directories required for the project structure
REQUIRED_DIRS = [
    os.path.join('training_data', 'raw'),
    os.path.join('training_data', 'cleaned'),
    'logs', 
    os.path.join('..', 'ai-microservice')
]
