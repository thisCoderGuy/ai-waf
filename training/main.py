import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging 
from datetime import datetime 
import time 

from loggers import global_logger, evaluation_logger

from config import (
    TEST_SIZE,
    REQUIRED_DIRS, 
    RANDOM_STATE
)
from data_loader import load_and_clean_data
from data_preprocessor import preprocess_data
from model_trainer import train_model
from model_evaluator import evaluate_model
from model_utils import save_model_and_preprocessor


def _create_required_directories(dirs):
    """Creates all specified directories if they do not exist."""
    global_logger.debug("Creating required directories...")
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        global_logger.debug(f"  Created/Ensured directory: {d}")

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
    

    
    evaluation_logger.info(f"####################################################################")
    evaluation_logger.info(f"Model Training Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directories if they don't exist
    _create_required_directories(REQUIRED_DIRS)

    # 1. Load and Clean Data
    df = load_and_clean_data()
    if df.empty:
        evaluation_logger.error("No data remaining after cleaning. Cannot proceed with training.")
        return
    global_logger.debug(f"{type(df)=}")
    global_logger.debug(f"{df.shape=}")
    global_logger.debug(f"{df.columns=}")

    # 2. Preprocess Data (Feature extraction) (No preprocessing necessary for deep learning as they use their own embedding layers)
    preprocessor, X, y = preprocess_data(df)
    global_logger.debug(f"{type(X)=}\n{type(y)=}")

    # Ensure there are enough samples and features after preprocessing
    if X.shape[0] == 0 or X.shape[1] == 0:
        evaluation_logger.error("Error: No features generated after preprocessing. Check your data and preprocessing steps.")
        return
    if len(y.unique()) < 2:
        evaluation_logger.warning("Only one class present in labels. Cannot perform classification.")
        return

    # Split data into training and testing sets
    # stratify=y ensures that both train and test sets have proportional class distributions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    global_logger.debug(f"\tTraining data shape: {X_train.shape}")
    global_logger.debug(f"\tTesting data shape: {X_test.shape}")    
    global_logger.debug(f"\tTraining labels: {y_train.shape}")
    global_logger.debug(f"\tTesting labels: {y_test.shape}")

    start_time = time.time()

    # 3. Train Model
    model = train_model(X_train, y_train, preprocessor) 
   
    # 4. Evaluate Model
    evaluate_model(model, X_test, y_test)

    end_time = time.time()
    total_duration = end_time - start_time
    evaluation_logger.info(f"\t---Total training and evaluation time ---\n\t{total_duration:.2f} seconds")

    # 5. Save Model and Preprocessor 
    save_model_and_preprocessor(model, preprocessor)

    
if __name__ == "__main__":
    main()
