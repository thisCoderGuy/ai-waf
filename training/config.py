import os

# --- Model Selection and Specific Parameters ---
# Set the current model type. 
# Possible values so far: 'svm', 'random_forest', 'decision_tree', 'naive_bayes', 'mlp',
#                         'fcnn', 'cnn', 'rnn', 'lstm', 'transformer', 'llm'  # For those choose N_SPLITS_CROSS_VALIDATION = small num
MODEL_TYPE = 'fcnn'

# --- General Model Training Configuration ---
# Test size vs training size
TEST_SIZE = 0.2
# Random state for reproducibility in data splitting and model training
RANDOM_STATE = 42

# --- Cross validation and hyperparameter tuning ---
# Cross validation and hyperparameter tuning:
PERFORM_TUNING = True
TUNING_METHOD = 'random' # 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
RANDOM_SEARCH_N_ITER = 10 # Number of parameter settings that are sampled if using RandomizedSearchCV
# Number of splits for Stratified K-Fold Cross-Validation (used by traditional models)
N_SPLITS_CROSS_VALIDATION = 3


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
    },
    'mlp': { 
        'hidden_layer_sizes': (64,), # One hidden layer with 64 neurons
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001, # L2 regularization parameter
        'learning_rate_init': 0.001,
        'max_iter': 200, # Number of epochs
        'verbose': False # Set to True for verbose training output
    },
    'fcnn': {
         'hidden_size': 64,
         'learning_rate': 0.001,
         'epochs': 50,
         'batch_size': 32
    },
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
    },
    'llm': {
        'model_name': 'gemini-2.0-flash', # Or other LLM models
        'temperature': 0.7,
        'max_tokens': 150,
        # Any other specific LLM parameters like fine-tuning settings, API keys etc.
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
    },
    'decision_tree': { 
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    },
    'naive_bayes': { 
        'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0] 
    },
    'mlp': {
        'hidden_layer_sizes': [(32,), (64,), (32, 32), (64, 32)], # Example: 1 or 2 hidden layers
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01], # L2 regularization strength
        'learning_rate_init': [0.0005, 0.001, 0.005],
        'max_iter': [100, 200, 300] # Number of epochs
    },
    'fcnn': {
         'hidden_size': [32, 64],
         'learning_rate': [0.001, 0.01],
         'epochs': [50, 100]
    }
}



# --- PyTorch Specific Configuration ---
OPTIMIZER_PARAMS = {
    'type': 'Adam', # Options: 'Adam', 'SGD', etc.
    'hyperparameters': {
        'lr': 0.001, # Default learning rate for Adam
        'weight_decay': 0, # regularization
        # Add other Adam-specific params here, e.g., 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0
    }
}

LOSS_PARAMS = {
    'type': 'CrossEntropyLoss', # Options: 'CrossEntropyLoss', 'BCEWithLogitsLoss', etc.
    'hyperparameters': {
        # Example for CrossEntropyLoss with class weights (adjust weights based on your class distribution)
        # 'weight': [1.0, 10.0] # Needs to be a torch.Tensor, converted inside the wrapper
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
