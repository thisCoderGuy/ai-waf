import os


#########################################################
# --- 1. Data Loading and Cleaning ---
#########################################################

# Do perform data cleaning?
PERFORM_DATA_CLEANING = True # True or False

           #########################################################
           # if data cleaning is performed
           #########################################################
# Which raw datasets to load?
# Path to raw dataset files (Coraza audit log files)
RAW_DATA_FILE_PATHS = [
    os.path.join('training', 'training-data', 'raw', 'coraza-audit-benign_20250710_200141.csv'),
    os.path.join('training', 'training-data', 'raw', 'coraza-audit-sqli_20250710_201920.csv'),
    os.path.join('training', 'training-data', 'raw', 'coraza-audit-xss_20250710_203049.csv'),
    os.path.join('training', 'training-data', 'raw', 'coraza-audit-dta_20250710_205839.csv'),
    os.path.join('training', 'training-data', 'raw', 'coraza-audit-csrf_20250710_210344.csv'),
    os.path.join('training', 'training-data', 'raw', 'coraza-audit-enum_20250710_210931.csv'),
]

# Features that must be present for each data point
CRITICAL_FEATURES = ['RequestURI', 'RequestMethod']

# Columns to drop during initial data cleaning
COLUMNS_TO_DROP = [
    'Timestamp', 'TransactionID', 'ClientIP', 'ClientPort', 'ServerIP', 'ServerPort',
    'ResponseProtocol', 'ResponseHeaders', 'ResponseBody',
    'WAFInterrupted', 'InterruptionRuleID', 'InterruptionStatus', 'InterruptionAction',
    'MatchedRulesCount', 'MatchedRulesIDs', 'MatchedRulesMessages', 'MatchedRulesTags',
    'AIScore', 'AIVerdict' # 'TimeOfDayHour', 'TimeOfDayDayOfWeek' if they were present
]

COLUMNS_TO_UPPERCASE = ['RequestMethod']

# Specific problematic endings to filter out from raw log lines
PROBLEMATIC_ENDINGS = [
    "Failed to write CSV record: short write",
    "CSV writer error: short write"
]

# Where to save the cleaned data? (A timestamp will be added to the provided filename)
# Path to save the cleaned dataset (within 'training-data' folder)
CLEANED_DATA_OUTPUT_PATH = os.path.join('training', 'training-data', 'cleaned', 'coraza-audit-cleaned.csv') 


           #########################################################
           # if data cleaning is NOT performed
           #########################################################

# Which cleaned dataset to load?
CLEANED_DATA_PATH = os.path.join('training', 'training-data', 'cleaned', 'coraza-audit-cleaned_20250709_181524.csv') 

#########################################################
# --- 2. Training ---
#########################################################
# Do perform training?
PERFORM_TRAINING = False # True or False


   #########################################################
   # --- 2.1. Preprocessing (Feature extraction) ---
   #########################################################

### Categorizing features
# Textual features
TEXT_FEATURES = [
            'RequestURIPath',
            'RequestURIQuery',
            'RequestBody',
            'UserAgent' 
]

# Categorical features
CATEGORICAL_FEATURES = ['RequestMethod']

# Numerical features
NUMERICAL_FEATURES = ['RequestLength', 'PathLength', 'QueryLength']

# Labels
LABEL = 'AIVerdictLabel'
LABEL_VALUES = ['benign', 'malicious']


# Sparse data features: features where most values are zero or missing, often high-dimensional data
# Dense data features: features where most values are non-zero.

# Traditional machine learning algorithms generally handle sparse data well. 
# Deep learning algorithms: Neural nets expect dense tensors.

TYPE_OF_PREPROCESSING = 'sparse' # 'sparse' or 'dense'



           #################################
           # If performing Sparse Preprocessing:
           #################################           

#### Textual features

# TF-IDF is a sparse vectorization method that provides	weighted counts	of terms based on their frequency in documents and rarity across the corpus.

# TfidfVectorizer Parameters for each feature
#   - Character-level tokenization splits text into single characters. It handles any text well, especially noisy or unknown words, but creates longer sequences and requires more training.
#   - Word-level tokenization splits text into whole words. It produces shorter sequences and is more intuitive but needs a large vocabulary and struggles with unknown words.

TFIDF_MAX_FEATURES = {
            'RequestURIPath': 5000,
            'RequestURIQuery': 5000,
            'RequestBody': 5000,
            'UserAgent': 5000,
        }
TFIDF_ANALYZERS = {
            'RequestURIPath': 'char', # 'word' or 'char'
            'RequestURIQuery': 'char', # 'word' or 'char'
            'RequestBody': 'char', # 'word' or 'char'
            'UserAgent': 'char', # 'word' or 'char'
        }
TFIDF_NGRAM_RANGES  = {
            'RequestURIPath': (1, 6), # For character n-grams
            'RequestURIQuery':  (1, 6), # For character n-grams
            'RequestBody':  (1, 6), # For character n-grams
            'UserAgent':  (1, 2), # For character n-grams
        }

#### Categorical features

# OneHotEncoder is a sparse encoding method.

#### Numerical features

# StandardScaler is a feature scaling method that standardizes features by removing the mean and scaling to unit variance, so that each feature has zero mean and unit standard deviation.

           #################################
           # If performing Dense Preprocessing:
           ################################# 

#### Textual features
# Deep Learning models will include their own embedding methods (that generate dense vectors in dim < vocabulary size like Word2Vec, BERT, etc.)
# so we just need to we just tokenize the textual features

# Tokenizer (from tensorflow.keras.preprocessing.text) provides integer sequences representing tokens.
#   - Character-level tokenization splits text into single characters. It handles any text well, especially noisy or unknown words, but creates longer sequences and requires more training.
#   - Word-level tokenization splits text into whole words. It produces shorter sequences and is more intuitive but needs a large vocabulary and struggles with unknown words.
TOKENIZER_CONFIGS = {
            'RequestURIPath': 'char', # 'word' or 'char'
            'RequestURIQuery': 'char', # 'word' or 'char'
            'RequestBody': 'char', # 'word' or 'char'
            'UserAgent': 'char', # 'word' or 'char'
        }
# Max sequences
MAX_SEQ_LENGTHS = {
            'RequestURIPath': 50,
            'RequestURIQuery': 50,
            'RequestBody': 50,
            'UserAgent': 100
        }

#### Categorical features

# LabelEncoder dense encoding method. It converts categorical labels into integer values.

#### Numerical features

# StandardScaler is a feature scaling method that standardizes features by removing the mean and scaling to unit variance, so that each feature has zero mean and unit standard deviation.


   #########################################################
   # --- 2.2. Model Selection and Specific Model Parameters ---
   #########################################################

# --- General Model Training Configuration ---
# Test size vs training size
TEST_SIZE = 0.2
# Random state for reproducibility in data splitting and model training
RANDOM_STATE = 42
# Set the current model type. 
# Possible values so far: 'svm', 'random_forest', 'decision_tree', 'naive_bayes', 'mlp',
#                         'fcnn', 'cnn', 'rnn', 'lstm', 'transformer', 'llm'  # For those choose N_SPLITS_CROSS_VALIDATION = small num
MODEL_TYPE = 'cnn'

MODEL_CLASSES = {
    'svm': 'SVC',
    'random_forest': 'RandomForestClassifier',
    'decision_tree': 'DecisionTreeClassifier',
    'naive_bayes': 'MultinomialNB',
    'mlp': 'MLPClassifier',
    'fcnn': 'PyTorchMLPClassifier',
    'cnn': 'CNNClassifier',
    'rnn': 'RNNClassifier'
}

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
        'numerical_hidden_size': 32,
        'text_embed_dims': {
            'RequestURIPath': 32,
            'RequestURIQuery': 32,
            'RequestBody': 32,
            'UserAgent': 32
        },
        'categorical_embed_dims': {
            'RequestMethod': 3
        },
        
    },    
    'cnn': { 
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
        'numerical_hidden_size': 32,
        'text_embed_dims': {
            'RequestURIPath': 32,
            'RequestURIQuery': 32,
            'RequestBody': 32,
            'UserAgent': 32
        },
        'categorical_embed_dims': {
            'RequestMethod': 3
        },
        'text_cnn_configs': {
            'RequestURIPath': {'n_filters': 32, 'filter_size': 3},
            'RequestURIQuery': {'n_filters': 48, 'filter_size': 5},
            'RequestBody': {'n_filters': 96, 'filter_size': 5},
            'UserAgent': {'n_filters': 64, 'filter_size': 4}
        },
        
    },
    'rnn': {
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
        'numerical_hidden_size': 32,
        'text_embed_dims': {
            'RequestURIPath': 32,
            'RequestURIQuery': 32,
            'RequestBody': 32,
            'UserAgent': 32
        },
        'categorical_embed_dims': {
            'RequestMethod': 3
        },
        'text_rnn_configs': {
            'RequestURIPath': {'hidden_size': 128, 'num_layers': 1, 'bidirectional': False},
            'RequestURIQuery': {'hidden_size': 128, 'num_layers': 1, 'bidirectional': False},
            'RequestBody': {'hidden_size': 128, 'num_layers': 1, 'bidirectional': False},
            'UserAgent': {'hidden_size': 128, 'num_layers': 1, 'bidirectional': False}
        },
        'rnn_type': 'GRU',  # LSTM or GRU
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
   # --- 2.3. Cross validation (CV) and hyperparameter tuning ---
   #########################################################


# In Traditional Machine Learning: 
# - Cross-validation is routinely used (k-fold cross-validation) to
#   - obtain a more reliable estimate of a model’s generalization performance.
#   - help reduce overfitting by evaluating on multiple train/test splits.
# - Hyperparameter tuning is often done using:
#   - Grid search or random search over parameter spaces.
#   - Often *combined* with cross-validation (e.g., GridSearchCV in scikit-learn).

# In Deep Learning:
# - Cross-validation is less commonly used due to High computational cost
#    - Instead, data is typically simply split into training, validation, and test sets.
#    - Early stopping on validation loss often replaces cross-validation for performance monitoring.
# - Hyperparameter tuning is still crucial: ### TODO ###
#    - Done via manual tuning, random search, or Bayesian optimization (e.g., Optuna, Ray Tune).
#    - Recent tools (like Keras Tuner, Hugging Face Optuna) automate the process.
#    - Can be more complex due to many more hyperparameters (e.g., learning rate, number of layers, batch size).

# Do perform Cross validation and hyperparameter tuning?
PERFORM_TUNING = False # True or False

           #########################################################
           # if CV and hyperparameter tuning is performed
           #########################################################

CV_AND_TUNING_METHOD = 'random' # 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
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
         'hidden_size': [32], # [32, 64],
         'learning_rate': [0.01], #0.001, 0.01],
    },
    'cnn': {
         'hidden_size': [32], # [32, 64],
         'learning_rate': [0.01], #0.001, 0.01],
    },
    'rnn': {
         'hidden_size': [32], # [32, 64],
         'learning_rate': [0.01], #0.001, 0.01],
    }
}




   #########################################################
   # --- 2.4. Model Output Configuration ---
   #########################################################
# Path to save the trained model.
MODEL_BASE_OUTPUT_DIR = os.path.join('ai-microservice', 'trained-models')
# Prefix for the generated model and preprocessor filenames (e.g., 'svm_malicious_traffic_model_20250611_1330.joblib')
MODEL_FILENAME_PREFIX = 'malicious_traffic_model'
PREPROCESSOR_FILENAME_PREFIX = 'malicious_traffic_preprocessor'

#########################################################
# --- Loggers Setup ---
#########################################################
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
        'file': {
            'class': 'logging.FileHandler',
            'filename': os.path.join('training', 'model-logs', 'all_models.txt'),  # <-- Log file
            'mode': 'a',
            'level': 'INFO',                        #  Keep level to INFO to log only important info to the file  
            'formatter': 'standard',
        },
    },

    'loggers': {
        'evaluation_logger': {
            'level': 'INFO',                            
            'handlers': ['console', 'file'],
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