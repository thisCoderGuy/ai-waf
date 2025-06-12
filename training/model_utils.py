import joblib
import os
import logging 

from datetime import datetime 

from config import (
    MODEL_TYPE, MODEL_FILENAME_PREFIX, PREPROCESSOR_FILENAME_PREFIX, MODEL_BASE_OUTPUT_DIR
)

def save_model_and_preprocessor(model, preprocessor, logger=None):
    """
    Saves the trained model and preprocessor to specified paths.

    Args:
        model (object): The trained machine learning model.
        preprocessor (object): The fitted ColumnTransformer preprocessor.
        model_path (str): The full file path to save the model.
        preprocessor_path (str): The full file path to save the preprocessor.
        logger (logging.Logger, optional): Logger object to write messages. Defaults to None.
    """
    current_model_type = MODEL_TYPE.lower()
    logger.info("--- Model Saving ---")   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{current_model_type}_{MODEL_FILENAME_PREFIX}_{timestamp}.joblib"
    preprocessor_filename = f"{current_model_type}_{PREPROCESSOR_FILENAME_PREFIX}_{timestamp}.joblib"

    model_output_path = os.path.join(MODEL_BASE_OUTPUT_DIR, model_filename)
    preprocessor_output_path = os.path.join(MODEL_BASE_OUTPUT_DIR, preprocessor_filename)

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_output_path), exist_ok=True)
    
    joblib.dump(model, model_output_path)
    joblib.dump(preprocessor, preprocessor_output_path)


    if logger:
        logger.info(f"\tSaved preprocessor to {preprocessor_output_path}...")
        logger.info(f"\tSaved trained model to {model_output_path}...")
    else:
        print(f"Saved preprocessor to {preprocessor_output_path}...")
        print(f"Saved trained model to {model_output_path}...")

def load_model_and_preprocessor(model_path, preprocessor_path, logger=None):
    """
    Loads a trained model and preprocessor from specified paths.

    Args:
        model_path (str): The file path to load the model from.
        preprocessor_path (str): The file path to load the preprocessor from.
        logger (logging.Logger, optional): Logger object to write messages. Defaults to None.

    Returns:
        tuple: A tuple containing the loaded model and preprocessor.
    """
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        if logger:
            logger.info("Model and preprocessor loaded successfully.")
        else:
            print("Model and preprocessor loaded successfully.")
        return model, preprocessor
    except FileNotFoundError:
        message = f"Error: Model or preprocessor file not found at {model_path} or {preprocessor_path}"
        if logger:
            logger.error(message)
        else:
            print(message)
        return None, None
    except Exception as e:
        message = f"Error loading model or preprocessor: {e}"
        if logger:
            logger.error(message)
        else:
            print(message)
        return None, None
