import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging 
from datetime import datetime 
import time 



from config import (
    TEST_SIZE,
    REQUIRED_DIRS, EVALUATION_LOG_PATH,
    RANDOM_STATE
)
from data_loader import load_and_clean_data
from data_preprocessor import preprocess_data
from model_trainer import train_model
from model_evaluator import evaluate_model
from model_utils import save_model_and_preprocessor

def _create_required_directories(dirs):
    """Creates all specified directories if they do not exist."""
    print("Creating required directories...")
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  Created/Ensured directory: {d}")

def _setup_logger(log_file_path):
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
    _create_required_directories(REQUIRED_DIRS)

    # Setup logger
    logger = _setup_logger(EVALUATION_LOG_PATH)
    logger.info(f"####################################################################")
    logger.info(f"Model Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load and Clean Data
    df = load_and_clean_data(logger)
    if df.empty:
        logger.warning("No data remaining after cleaning. Cannot proceed with training.")
        return

    # 2. Preprocess Data (Feature extraction) (No preprocessing necessary for deep learning as they use their own embedding layers)
    preprocessor, X, y = preprocess_data(df, logger)

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

    logger.info(f"\tTraining data shape: {X_train.shape}")
    logger.info(f"\tTesting data shape: {X_test.shape}")

    start_time = time.time()

    # 3. Train Model
    model = train_model(X_train, y_train, logger) 
    
    # 4. Evaluate Model
    evaluate_model(model, X_test, y_test, logger)

    end_time = time.time()
    total_duration = end_time - start_time
    logger.info(f"\t---Total training and evaluation time ---\n\t{total_duration:.2f} seconds")

    # 5. Save Model and Preprocessor 
    save_model_and_preprocessor(model, preprocessor, logger)

    
if __name__ == "__main__":
    main()
