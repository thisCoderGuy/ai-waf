import joblib
import os
import logging # Import logging

def save_model_and_preprocessor(model, preprocessor, model_path, preprocessor_path, logger=None):
    """
    Saves the trained model and preprocessor to specified paths.

    Args:
        model (object): The trained machine learning model.
        preprocessor (object): The fitted ColumnTransformer preprocessor.
        model_path (str): The full file path to save the model.
        preprocessor_path (str): The full file path to save the preprocessor.
        logger (logging.Logger, optional): Logger object to write messages. Defaults to None.
    """
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

    if logger:
        logger.info(f"\nSaving trained model to {model_path}...")
    else:
        print(f"\nSaving trained model to {model_path}...")
    joblib.dump(model, model_path)

    if logger:
        logger.info(f"Saving preprocessor to {preprocessor_path}...")
    else:
        print(f"Saving preprocessor to {preprocessor_path}...")
    joblib.dump(preprocessor, preprocessor_path)

    if logger:
        logger.info("Model and preprocessor saved successfully.")
    else:
        print("Model and preprocessor saved successfully.")

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
