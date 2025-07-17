import joblib
import os
import logging 

from datetime import datetime 

from training_config import (
    MODEL_TYPE, MODEL_FILENAME_PREFIX, PREPROCESSOR_FILENAME_PREFIX, 
    MODEL_BASE_OUTPUT_DIR, LATEST_MODEL_INFO_PATH
)

def save_model_and_preprocessor(model, preprocessor, logger=None):
    """
    Saves the trained model and preprocessor to specified paths.
    Also updates a file indicating the path of the latest trained model for the microservice.


    Args:
        model (object): The trained machine learning model.
        preprocessor (object): The fitted ColumnTransformer preprocessor.
        logger (logging.Logger, optional): Logger object to write messages. Defaults to None.
    """
    if logger:
        logger.info("--- Model Saving ---")   

    current_model_type = MODEL_TYPE.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{current_model_type}_{MODEL_FILENAME_PREFIX}_{timestamp}.joblib"
    model_output_path = os.path.join(MODEL_BASE_OUTPUT_DIR, model_filename)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    try:
        joblib.dump(model, model_output_path)
        
        if logger:
            logger.info(f"\tSaved trained model to {model_output_path}...")
        else:
            print(f"Saved trained model to {model_output_path}...")

        # Save the preprocessor only if it's not None
        preprocessor_filename = None # Initialize to None
        if preprocessor is not None:
            preprocessor_filename = f"{current_model_type}_{PREPROCESSOR_FILENAME_PREFIX}_{timestamp}.joblib"
            preprocessor_output_path = os.path.join(MODEL_BASE_OUTPUT_DIR, preprocessor_filename)
            joblib.dump(preprocessor, preprocessor_output_path)
            if logger:
                logger.info(f"\tSaved preprocessor to {preprocessor_output_path}...")
            else:
                print(f"Saved preprocessor to {preprocessor_output_path}...")
        else:
            if logger:
                logger.info("\tNo preprocessor provided. Skipping preprocessor saving.")
            else:
                print("No preprocessor provided. Skipping preprocessor saving.")

        try:
            # Ensure the directory for LATEST_MODEL_INFO_PATH exists
            os.makedirs(os.path.dirname(LATEST_MODEL_INFO_PATH), exist_ok=True)
            
            with open(LATEST_MODEL_INFO_PATH, 'w') as f:
                # Store the model filename and optionally the preprocessor filename
                f.write(f"model_filename={model_filename}\n")
                if preprocessor_filename:
                    f.write(f"preprocessor_filename={preprocessor_filename}\n")
            if logger:
                logger.info(f"\tUpdated latest model info in {LATEST_MODEL_INFO_PATH}")
            else:
                print(f"Updated latest model info in {LATEST_MODEL_INFO_PATH}")
        except Exception as e:
            message = f"Error updating latest model info file {LATEST_MODEL_INFO_PATH}: {e}"
            if logger:
                logger.error(message)
            else:
                print(message)
        # --- END NEW LOGIC ---

    except Exception as e:
        message = f"Error saving model or preprocessor: {e}"
        if logger:
            logger.error(message)
        else:
            print(message)

def load_model_and_preprocessor(model_path, preprocessor_path, logger=None):
    """
    Loads a trained model and preprocessor from specified paths.

    Args:
        model_path (str): The file path to load the model from.
        preprocessor_path (str, optional): The file path to load the preprocessor from. 
                                           Defaults to None. If None, only the model is loaded.
        logger (logging.Logger, optional): Logger object to write messages. Defaults to None.

    Returns:
         tuple: A tuple containing the loaded model and preprocessor (or None if not loaded).
    """
    model = None
    preprocessor = None
    try:
        model = joblib.load(model_path)
        if logger:
            logger.info("Model loaded successfully.")
        else:
            print("Model loaded successfully.")

        if preprocessor_path:
            preprocessor = joblib.load(preprocessor_path)
            if logger:
                logger.info("Preprocessor loaded successfully.")
            else:
                print("Preprocessor loaded successfully.")
        else:
            if logger:
                logger.info("No preprocessor path provided. Skipping preprocessor loading.")
            else:
                print("No preprocessor path provided. Skipping preprocessor loading.")

        return model, preprocessor
    except FileNotFoundError as e:
        message = f"Error: File not found. {e}. Please check the paths: Model path: {model_path}, Preprocessor path: {preprocessor_path}"
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
