import os

#########################################################
# --- Logger Setup ---
#########################################################
# Path for the  log file
EVALUATION_LOG_PATH = os.path.join('model-logs', 'all_models.txt')


#########################################################
# --- Directory Setup (created if they don't exist) ---
#########################################################
# Directories required for the project structure
REQUIRED_DIRS = [
    os.path.join('training-data', 'raw'),
    os.path.join('training-data', 'cleaned'),
    'model-logs', 
    os.path.join('..', 'ai-microservice', 'trained-models')
]

#########################################################
# --- Data Loading and Cleaning ---
#########################################################
# Do perform data cleaning?
PERFORM_DATA_CLEANING = False # True or False

# Path to raw dataset files (Coraza audit log files)
LOG_FILE_PATHS = [
    os.path.join('training-data', 'raw', 'coraza-audit-benign.csv'),
    os.path.join('training-data', 'raw', 'coraza-audit-sqli.csv'),
    os.path.join('training-data', 'raw', 'coraza-audit-xss.csv')
]

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

# Path to save the cleaned dataset (within 'training-data' folder)
CLEANED_DATA_OUTPUT_PATH = os.path.join('training-data', 'cleaned', 'coraza-audit-cleaned.csv')


#########################################################
# --- Preprocessing (Feature extraction) ---
#########################################################
#################################
# Do perform Sparse Preprocessing?
# True for traditional machine learning algorithms (generate sparse, high dimensional vectors)
# False for deep learning algorithms because they will include their own embedding methods (that generate dense vectors in dim < vocabulary size)
PERFORM_SPARSE_PREPROCESSING = False # True or False

# --- Feature Extraction Parameters (for TfidfVectorizer) for text columns ---
TFIDF_MAX_FEATURES = 5000
# URI_PATH column
URI_PATH_TFIDF_ANALYZER = 'char' # 'word' or 'char'
URI_PATH_TFIDF_NGRAM_RANGE = (1, 6) # For character n-grams
# URI_QUERY column
URI_QUERY_TFIDF_ANALYZER = 'char' # 'word' or 'char'
URI_QUERY_TFIDF_NGRAM_RANGE = (1, 6) # For character n-grams
# BODY column
BODY_TFIDF_ANALYZER = 'char' # 'word' or 'char'
BODY_TFIDF_NGRAM_RANGE = (1, 6) # For character n-grams
# USER_AGENT column
USER_AGENT_TFIDF_ANALYZER = 'word' # 'word' or 'char'
USER_AGENT_TFIDF_NGRAM_RANGE = (1, 2) # For character n-grams

#################################
# Do perform Dense Preprocessing?
# True for  deep learning algorithms 
PERFORM_DENSE_PREPROCESSING = True
#Using character-level tokenization for text features
# Max sequences
MAX_SEQ_LENGTH_PATH = 100
MAX_SEQ_LENGTH_QUERY = 100
MAX_SEQ_LENGTH_BODY = 100
MAX_SEQ_LENGTH_USER_AGENT = 100



#########################################################
# --- Model Selection and Specific Parameters ---
#########################################################
# --- General Model Training Configuration ---
# Test size vs training size
TEST_SIZE = 0.2
# Random state for reproducibility in data splitting and model training
RANDOM_STATE = 42
# Set the current model type. 
# Possible values so far: 'svm', 'random_forest', 'decision_tree', 'naive_bayes', 'mlp',
#                         'fcnn', 'cnn', 'rnn', 'lstm', 'transformer', 'llm'  # For those choose N_SPLITS_CROSS_VALIDATION = small num
MODEL_TYPE = 'fcnn'

# --- Model Parameters  ---
MODEL_PARAMS = {
    ###########################################
    ### Traditional Machine Learning Algorithms
    ###########################################
    'svm': {
        'kernel': 'linear',
        'probability': True,
        'C': 1.0 # Regularization parameter, example
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': None, # None means nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    },
    'decision_tree': {
        'max_depth': None,
    },
    'naive_bayes': {
        # No specific parameters commonly tuned for MultinomialNB or GaussianNB, depending on data
    },
    'mlp': {  # mlp from scikitlearn
        'hidden_layer_sizes': (64,), # One hidden layer with 64 neurons
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001, # L2 regularization parameter
        'learning_rate_init': 0.001,
        'max_iter': 200, # Number of epochs
        'verbose': False # Set to True for verbose training output
    },
    ###########################################
    ### Deep Learning Algorithms
    ###########################################
    'fcnn': { # mlp from pytorch
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 32,
        #'activation': 'relu',
        'optimizer_type': 'adam', # adam or sgd
        'optimizer_params': {
            'weight_decay': 0.0001, # regularization
            # Add other Adam-specific params here, e.g., 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0
        },
        'loss_type':  'CrossEntropyLoss', # Options: 'CrossEntropyLoss', 'BCEWithLogitsLoss', etc.
        'loss_params': {
            # Example for CrossEntropyLoss with class weights (adjust weights based on your class distribution)
            # 'weight': [1.0, 10.0] # Needs to be a torch.Tensor, converted inside the wrapper
        },
        'dropout_rate': 0.5, 
        'hidden_size': 64,
        'num_classes': 2,
        'text_embed_dim': 32,
        'numerical_hidden_size': 32,
        'categorical_embedding_dim': 2,
        
    },
    'cnn': {
        'learning_rate': 0.001,
        'epochs': 10,
        'batch_size': 32,
        #'activation': 'relu',
        'optimizer_type': 'adam', # adam or sgd
        'optimizer_params': {
            'weight_decay': 0.0001, # regularization
            # Add other Adam-specific params here, e.g., 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0
        },
        'loss_type':  'CrossEntropyLoss', # Options: 'CrossEntropyLoss', 'BCEWithLogitsLoss' <=== if binary classification with 1 logit output, etc.
        'loss_params': {
            # Example for CrossEntropyLoss with class weights (adjust weights based on your class distribution)
            # 'weight': [1.0, 10.0] # Needs to be a torch.Tensor, converted inside the wrapper
        },
        'n_filters': 128,
        'filter_size': 5,
        'pool_size': 2,
        'dense_units': 64
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

#########################################################
# --- Cross validation and hyperparameter tuning ---
#########################################################
# Do perform Cross validation and hyperparameter tuning?
PERFORM_TUNING = True # True or False
TUNING_METHOD = 'random' # 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
RANDOM_SEARCH_N_ITER = 10 # Number of parameter settings that are sampled if using RandomizedSearchCV
# Number of splits for Stratified K-Fold Cross-Validation 
N_SPLITS_CROSS_VALIDATION = 3

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
    }
}




#########################################################
# --- Model Output Configuration ---
#########################################################
# Path to save the trained model.
MODEL_BASE_OUTPUT_DIR = os.path.join('..', 'ai-microservice', 'trained-models')
# Prefix for the generated model and preprocessor filenames (e.g., 'svm_malicious_traffic_model_20250611_1330.joblib')
MODEL_FILENAME_PREFIX = 'malicious_traffic_model'
PREPROCESSOR_FILENAME_PREFIX = 'malicious_traffic_preprocessor'
