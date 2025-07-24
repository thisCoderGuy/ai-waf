import joblib
import os
import logging
import torch
import json
from datetime import datetime



from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline 
from sklearn.neural_network import MLPClassifier 

from MLP_wrapper import PyTorchMLPClassifier 
from CNN_wrapper import CNNClassifier
from RNN_wrapper import RNNClassifier




from training_config import (
    MODEL_ARCHITECTURE, MODEL_FILENAME_PREFIX, PREPROCESSOR_FILENAME_PREFIX, TRAINING_TYPE,
    MODEL_BASE_OUTPUT_DIR, LATEST_MODEL_INFO_PATH, MODEL_CLASSES
)

def save_model_and_preprocessor(model, preprocessor, model_params_to_save: dict = None, logger=None):
    """
    Saves the trained model and preprocessor to specified paths.
    Also updates a file indicating the path of the latest trained model for the microservice.
    Handles both traditional ML (joblib) and Deep Learning (PyTorch) models.


    Args:
        model (object): The trained machine learning model (scikit-learn or PyTorch).
        preprocessor (object): The fitted preprocessor (typically scikit-learn compatible).
        logger (logging.Logger, optional): Logger object to write messages. Defaults to None.
  
    """
    if logger:
        logger.info("--- Model Saving ---")   

    current_training_type = TRAINING_TYPE.lower() # 'traditional' or 'deep'

    current_model_type = MODEL_ARCHITECTURE.lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a unique directory for this model version's artifacts
    model_version_dir = os.path.join(MODEL_BASE_OUTPUT_DIR, f"{current_model_type}_{timestamp}")
    os.makedirs(model_version_dir, exist_ok=True)

    # Determine model filename based on type
    model_extension = ".joblib"
    if current_model_type == 'deep':
        model_extension = ".pt" # PyTorch models typically use .pt or .pth
    
    model_filename = f"{current_training_type}_{current_model_type}_{MODEL_FILENAME_PREFIX}{model_extension}" 
    model_output_path = os.path.join(model_version_dir, model_filename)

    
       
    try:
        if current_training_type == 'traditional':
            joblib.dump(model, model_output_path)
        elif current_training_type == 'deep':
            torch.save(model.model.state_dict(), model_output_path)
        else:
            raise ValueError(f"Unsupported TRAINING_TYPE: {TRAINING_TYPE}. Must be 'traditional' or 'deep'.")

        
        if logger:
            logger.info(f"\tSaved trained model to {model_output_path}...")
        else:
            print(f"Saved trained model to {model_output_path}...")

        # Save the preprocessor only if it's not None
        preprocessor_filename = None # Initialize to None
        if preprocessor is not None:
            preprocessor_filename = f"{current_training_type}_{current_model_type}_{PREPROCESSOR_FILENAME_PREFIX}.joblib" 
            preprocessor_output_path = os.path.join(model_version_dir, preprocessor_filename)
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


        # --- Save model_info.txt within the versioned directory ---
        model_info_filepath = os.path.join(model_version_dir, "model_info.txt")
        try:
            with open(model_info_filepath, 'w') as f:
                f.write(f"model_filename={model_filename}\n") # Filename relative to its versioned dir
                if preprocessor_filename:
                    f.write(f"preprocessor_filename={preprocessor_filename}\n") # Filename relative to its versioned dir
                f.write(f"model_type={current_model_type}\n")
                f.write(f"training_type={current_training_type}\n")
                f.write(f"timestamp={timestamp}\n")

                if current_training_type == 'deep' and model_params_to_save is not None:
                    try:
                        f.write(f"model_params={json.dumps(model_params_to_save)}\n")
                    except TypeError as json_err:
                        logging.error(f"Failed to serialize model_params to JSON: {json_err}. Skipping saving model_params.")
                        if logger:
                            logger.error(f"Failed to serialize model_params to JSON: {json_err}. Skipping saving model_params.", exc_info=True)
                        else:
                            print(f"Failed to serialize model_params to JSON: {json_err}. Skipping saving model_params.")
                            import traceback
                            traceback.print_exc()

            if logger:
                logger.info(f"\tSaved model metadata to {model_info_filepath}")
            else:
                print(f"Saved model metadata to {model_info_filepath}")
        except Exception as e:
            message = f"Error saving model_info.txt to {model_info_filepath}: {e}"
            if logger:
                logger.error(message, exc_info=True)
            else:
                print(message)
                import traceback
                traceback.print_exc()

        # --- Update LATEST_MODEL_INFO_PATH (pointing to the versioned directory) ---
        try:
            # Ensure the directory for LATEST_MODEL_INFO_PATH exists
            os.makedirs(os.path.dirname(LATEST_MODEL_INFO_PATH), exist_ok=True)
            
            with open(LATEST_MODEL_INFO_PATH, 'w') as f:
                # Store the latest model directory
                f.write(f"latest_model_version_dir={os.path.basename(model_version_dir)}\n")
                
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

    except Exception as e:
        message = f"Error saving model or preprocessor: {e}"
        if logger:
            logger.error(message)
        else:
            print(message)

def load_model_and_preprocessor(model_path: str, preprocessor_path: str = None, training_type: str = 'traditional', model_type_name: str = None, model_params: dict = None, logger=None):
    """
    Loads a trained model and preprocessor from specified paths, handling different model types.

    Args:
        model_path (str): The file path to load the model from.
        preprocessor_path (str, optional): The file path to load the preprocessor from.
                                           Defaults to None. If None, only the model is loaded.
        training_type (str): The type of training ('traditional' for scikit-learn, 'deep' for PyTorch).
                              This should be read from model_info.txt by app.py.
        model_type_name (str, optional): The name of the model architecture (e.g., 'cnn', 'rnn').
                                         Used primarily for PyTorch model instantiation.
                                         This should be read from model_info.txt by app.py.
        model_params (dict, optional): Dictionary of parameters required to instantiate the PyTorch model.
                                       This should be read from model_info.txt or a separate config file
                                       by app.py and passed here.
        logger (logging.Logger, optional): Logger object to write messages. Defaults to None.

    Returns:
         tuple: A tuple containing the loaded model and preprocessor (or None if not loaded).
    """
    model = None
    preprocessor = None
    try:
        if training_type == 'traditional':
            model = joblib.load(model_path)
        elif training_type == 'deep':
            if model_type_name is None:
                raise ValueError("model_type_name must be provided for deep learning models.")
            if model_params is None:
                logging.warning("No model_params provided for deep learning model. Attempting to instantiate without specific parameters. This might lead to errors.")
                model_params = {} # Default to empty dict if not provided

            # --- Dynamically get the PyTorch model class ---
            model_class_str = MODEL_CLASSES.get(model_type_name.lower())
            if model_class_str is None:
                raise ValueError(f"Unknown PyTorch model type: '{model_type_name}'. Not found in PYTORCH_MODEL_CLASS_MAP.")

            try:
                # Use globals() to get the class object from its string name
                ModelClass = globals()[model_class_str]
            except KeyError:
                raise ImportError(f"PyTorch model class '{model_class_str}' for model type '{model_type_name}' is not imported or defined in the current scope.")

            # --- Instantiate the model with appropriate parameters ---
            try:
                model = ModelClass(**model_params)
            except TypeError as e:
                raise TypeError(f"Error instantiating PyTorch model '{model_class_str}' with params {model_params}: {e}. "
                                "Ensure correct parameters are passed for this model type and that they are compatible with the model's __init__ signature.")

            # Load the state_dict into the instantiated model
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) 
            model.eval() # Set model to evaluation mode (disables dropout, batch norm updates)
            if logger:
                logger.info(f"PyTorch '{model_type_name}' model loaded successfully.")
            else:
                print(f"PyTorch '{model_type_name}' model loaded successfully.")
        else:
            raise ValueError(f"Unknown training_type '{training_type}'. Expected 'traditional' or 'deep'.")

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
            logger.error(message, exc_info=True)
        else:
            print(message)
            import traceback
            traceback.print_exc()
        return None, None
    except Exception as e:
        message = f"Error loading model or preprocessor: {e}"
        if logger:
            logger.error(message, exc_info=True)
        else:
            print(message)
            import traceback
            traceback.print_exc()
        return None, None

