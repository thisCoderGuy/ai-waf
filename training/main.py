import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging 
from datetime import datetime 



from config import (
    LOG_FILE_PATHS, COLUMNS_TO_DROP, PROBLEMATIC_ENDINGS, CLEANED_DATA_OUTPUT_PATH,    
    MODEL_BASE_OUTPUT_DIR, MODEL_FILENAME_PREFIX, PREPROCESSOR_FILENAME_PREFIX,
    N_SPLITS_CROSS_VALIDATION,
    RANDOM_STATE, PERFORM_TUNING, TUNING_METHOD, RANDOM_SEARCH_N_ITER, TEST_SIZE, TFIDF_MAX_FEATURES, TFIDF_ANALYZER, TFIDF_NGRAM_RANGE,
    REQUIRED_DIRS, EVALUATION_LOG_PATH,
    MODEL_TYPE,   
)
from data_loader import load_and_clean_data
from feature_extractor import preprocess_data
from model_trainer import train_model
from model_evaluator import evaluate_model
from model_utils import save_model_and_preprocessor

def create_required_directories(dirs):
    """Creates all specified directories if they do not exist."""
    print("Creating required directories...")
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  Created/Ensured directory: {d}")

def setup_logger(log_file_path):
    """Sets up a logger to output to console and a file."""
    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('evaluation_logger')
    logger.setLevel(logging.INFO)
    logger.propagate = False # Prevent messages from being passed to the root logger

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s') # No timestamp in console as it's added in the message
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter) # Use the same formatter, timestamp will be in the message

    # Add handlers to the logger
    # Only add handlers if they are not already present to avoid duplicates on multiple runs
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        logger.addHandler(console_handler)
    if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        logger.addHandler(file_handler)

    print(f"Logger set up. Evaluation metrics will be saved to {log_file_path}")
    return logger

def main():
    """
    Orchestrates the entire machine learning pipeline:
    1. Creates necessary directories.
    2. Sets up the logging system.
    3. Loads and cleans raw data.
    4. Preprocesses data to extract features.
    5. Splits data into training and testing sets.
    6. Trains the model using cross-validation.
    7. Evaluates the trained model and logs results.
    8. Saves the trained model and preprocessor.
    """
    # Create output directories if they don't exist
    create_required_directories(REQUIRED_DIRS)

    # Setup logger
    logger = setup_logger(EVALUATION_LOG_PATH)

    # 1. Load and Clean Data
    df = load_and_clean_data(LOG_FILE_PATHS, COLUMNS_TO_DROP, PROBLEMATIC_ENDINGS)

    if df.empty:
        logger.warning("No data remaining after cleaning. Cannot proceed with training.")
        return

    # Save the cleaned data to a new CSV file
    try:
        df.to_csv(CLEANED_DATA_OUTPUT_PATH, index=False)
        logger.info(f"\nCleaned data saved to {CLEANED_DATA_OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Error saving cleaned data to CSV: {e}")

    # 2. Preprocess Data (Feature extraction)
    preprocessor, X, y = preprocess_data(df, TFIDF_MAX_FEATURES, TFIDF_ANALYZER, TFIDF_NGRAM_RANGE)

    # Ensure there are enough samples and features after preprocessing
    if X.shape[0] == 0 or X.shape[1] == 0:
        logger.error("Error: No features generated after preprocessing. Check your data and preprocessing steps.")
        return
    if len(y.unique()) < 2:
        logger.warning("Only one class present in labels. Cannot perform classification.")
        logger.warning("Ensure your dummy data or real data has both 'benign' and 'malicious' examples.")
        return

    # Split data into training and testing sets
    # stratify=y ensures that both train and test sets have proportional class distributions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Testing data shape: {X_test.shape}")

    # 3. Train Model
    # Determine which model type to train based on config.py
    current_model_type = MODEL_TYPE.lower() # Ensure lowercase for consistent dictionary keys

    # Train the model
    model = train_model(X_train, y_train, model_type=current_model_type,
                        n_splits=N_SPLITS_CROSS_VALIDATION,
                        random_state=RANDOM_STATE,
                        perform_tuning=PERFORM_TUNING,
                        tuning_method=TUNING_METHOD,
                        random_search_n_iter=RANDOM_SEARCH_N_ITER) 
    # 4. Evaluate Model
    evaluation_config = {
        'model_type': current_model_type.upper(), # Log the model type in uppercase
        'n_splits_cv': N_SPLITS_CROSS_VALIDATION,
        'random_state': RANDOM_STATE,
        'test_data_points': X_test.shape[0] # Number of samples in the test set
    }
    evaluate_model(model, X_test, y_test, logger, evaluation_config)

    # 5. Save Model and Preprocessor    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{current_model_type}_{MODEL_FILENAME_PREFIX}_{timestamp}.joblib"
    preprocessor_filename = f"{current_model_type}_{PREPROCESSOR_FILENAME_PREFIX}_{timestamp}.joblib"

    model_output_path = os.path.join(MODEL_BASE_OUTPUT_DIR, model_filename)
    preprocessor_output_path = os.path.join(MODEL_BASE_OUTPUT_DIR, preprocessor_filename)
    save_model_and_preprocessor(model, preprocessor, model_output_path, preprocessor_output_path, logger)

if __name__ == "__main__":
    main()
