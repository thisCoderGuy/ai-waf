from flask import Flask, request, jsonify
import logging
import joblib
import json
import pandas as pd
import os

from config import (
     MODEL_PATH, PREPROCESSOR_PATH, # Existing explicit paths (can be used as fallback/initial defaults)
    BASE_MODEL_DIR, LATEST_MODEL_INFO_PATH, # Paths for dynamic loading
    TRAINING_TYPE, MODEL_TYPE
)

app = Flask(__name__)

# Configure logging to stdout/stderr which Docker captures
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Global variables to hold the loaded model and preprocessor
model = None
preprocessor = None

def load_ml_assets():
    global model, preprocessor
   
    current_model_path = None
    current_preprocessor_path = None
    
    training_type_from_info = None
    model_type_name_from_info = None 
    model_params_from_info = {} 
    
    logging.info("Attempting to load ML assets...")

    try:
        # --- Attempt to load from LATEST_MODEL_INFO_PATH first ---
        latest_model_version_dir = None
        if os.path.exists(LATEST_MODEL_INFO_PATH):
            logging.info(f"Reading latest model info pointer from {LATEST_MODEL_INFO_PATH}")
            with open(LATEST_MODEL_INFO_PATH, 'r') as f:
                for line in f:
                    if line.startswith("latest_model_version_dir="):
                        latest_model_version_dir = line.split('=', 1)[1].strip()
                        break
            
            if latest_model_version_dir:
                model_version_full_path = os.path.join(BASE_MODEL_DIR, latest_model_version_dir)
                model_info_filepath = os.path.join(model_version_full_path, "model_info.txt")

                if os.path.exists(model_info_filepath):
                    logging.info(f"Reading model metadata from {model_info_filepath}")
                    model_filename_from_info = None
                    preprocessor_filename_from_info = None
                    with open(model_info_filepath, 'r') as f:
                        for line in f:
                            if line.startswith("model_filename="):
                                model_filename_from_info = line.split('=', 1)[1].strip()
                            elif line.startswith("preprocessor_filename="):
                                preprocessor_filename_from_info = line.split('=', 1)[1].strip()
                            elif line.startswith("training_type="):
                                training_type_from_info = line.split('=', 1)[1].strip()
                            elif line.startswith("model_type="):
                                model_type_name_from_info = line.split('=', 1)[1].strip()
                            elif line.startswith("model_params="):
                                try:
                                    params_str = line.split('=', 1)[1].strip()
                                    model_params_from_info = json.loads(params_str)
                                except json.JSONDecodeError as json_err:
                                    logging.error(f"Failed to parse model_params from '{model_info_filepath}': {json_err}. Using empty params.")
                                    model_params_from_info = {}
                    
                    if model_filename_from_info and training_type_from_info and model_type_name_from_info:
                        current_model_path = os.path.join(model_version_full_path, model_filename_from_info)
                        current_preprocessor_path = os.path.join(model_version_full_path, preprocessor_filename_from_info) if preprocessor_filename_from_info else None
                        logging.info(f"Determined latest model: {current_model_path} (Training Type: {training_type_from_info}, Model Type: {model_type_name_from_info})")
                    else:
                        logging.warning(f"Metadata in '{model_info_filepath}' is incomplete. Falling back to explicit paths.")
                else:
                    logging.warning(f"Model info file '{model_info_filepath}' not found within version directory. Falling back to explicit paths.")
            else:
                logging.warning(f"'{LATEST_MODEL_INFO_PATH}' found but no 'latest_model_version_dir' entry. Falling back to explicit paths.")
        else:
            logging.warning(f"'{LATEST_MODEL_INFO_PATH}' not found. Falling back to explicit paths from config.")

        # --- Fallback to explicit paths if latest info couldn't be used ---
        if current_model_path is None:
            current_model_path = MODEL_PATH 
            current_preprocessor_path = PREPROCESSOR_PATH
            training_type_from_info = TRAINING_TYPE
            model_type_name_from_info = MODEL_TYPE
            model_params_from_info = {} # Default to empty if falling back
            logging.info(f"Using explicit model paths from config: Model={current_model_path}, Preprocessor={current_preprocessor_path}")

        # --- Load the model and preprocessor using the determined paths and types ---
        if current_model_path and os.path.exists(current_model_path):
            from ..training import model_utils # Ensure this import is correct relative to your app.py
            model, preprocessor = model_utils.load_model_and_preprocessor(
                model_path=current_model_path,
                preprocessor_path=current_preprocessor_path,
                training_type=training_type_from_info,
                model_type_name=model_type_name_from_info,
                model_params=model_params_from_info, # NEW: Pass the parsed model parameters
                logger=app.logger
            )
            
            if model:
                logging.info(f"ML assets loaded successfully using: Model={current_model_path}, Preprocessor={current_preprocessor_path}")
            else:
                logging.error("Failed to load model and/or preprocessor from determined paths.")
        else:
            logging.error(f"Model file not found at the determined path: {current_model_path}")
            model = None
            preprocessor = None

    except Exception as e:
        logging.error(f"Error loading ML assets: {e}", exc_info=True)
        model = None
        preprocessor = None
            


# Load ML assets when the Flask app starts
with app.app_context():
    load_ml_assets()

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    if not data:
        logging.warning("No JSON data provided in request.")
        return jsonify({"error": "No JSON data provided"}), 400
    
    # Extract raw features from the incoming JSON
    requestURIPath = data.get("RequestURIPath", '')
    queryLength =data.get("QueryLength", '')
    userAgent = data.get("UserAgent", '')
    requestLength = data.get("RequestLength", '')
    requestURIQuery = data.get("RequestURIQuery", '')
    pathLength = data.get("PathLength", '')
    requestMethod = data.get("RequestMethod", '')
    requestBody = data.get("RequestBody" , '')

    logging.info(f"Received request for classification: Method={requestMethod}, Path={requestURIPath}, Query={requestURIQuery}, Body={requestBody[:100]},  userAgent={userAgent}...") # Log first 100 chars of body

    verdict = "benign"
    score = 0.0 # Default score

    # Check if model and preprocessor are loaded
    if model is None or preprocessor is None:
        logging.error("ML model or preprocessor not loaded. Cannot classify.")
        return jsonify({"error": "AI service not fully initialized"}), 500

    try:
        # Create a Pandas DataFrame from the incoming request data
        # Ensure column names match those expected by the preprocessor
        
        input_df = pd.DataFrame([{
            'RequestMethod': requestMethod, 
            'RequestURIPath': requestURIPath, 
            'RequestURIQuery': requestURIQuery, 
            'RequestBody': requestBody, 
            'UserAgent': userAgent, 
            'RequestLength': requestLength, 
            'PathLength': pathLength, 
            'QueryLength': queryLength,
            }])

        
        # Preprocess the input data using the loaded preprocessor
        # Ensure fillna is applied to text columns as in preprocess_data
        input_df['RequestMethod'] = input_df['RequestMethod'].fillna('')
        input_df['RequestURIPath'] = input_df['RequestURIPath'].fillna('')
        input_df['RequestURIQuery'] = input_df['RequestURIQuery'].fillna('')
        input_df['RequestBody'] = input_df['RequestBody'].fillna('')
        input_df['UserAgent'] = input_df['UserAgent'].fillna('')


        # Transform the DataFrame into numerical features
        # We use .transform() here, NOT .fit_transform(), as the preprocessor
        # was already fitted during training.
        processed_input = preprocessor.transform(input_df)

        # Make prediction
        # model.predict returns the class label (0 or 1)
        prediction_label = model.predict(processed_input)[0]

        # model.predict_proba returns probabilities for each class
        # It returns a 2D array [[prob_class_0, prob_class_1]]
        prediction_proba = model.predict_proba(processed_input)[0]

        # Determine verdict and score based on prediction
        if prediction_label == 1: # Assuming 1 corresponds to 'malicious'
            verdict = "malicious"
            score = prediction_proba[1] # Probability of being malicious
        else:
            verdict = "benign"
            score = 1 - prediction_proba[1] # Probability of being benign (or 1 - prob_malicious)

        logging.info(f"Classification result: Verdict={verdict}, Score={score:.4f}")

    except Exception as e:
        logging.error(f"Error during classification: {e}", exc_info=True)
        # Fallback to benign in case of classification error
        verdict = "benign"
        score = 0.0
        return jsonify({"error": f"Internal classification error: {e}"}), 500

    return jsonify({
        "verdict": verdict,
        "score": float(score) # Ensure score is a standard float for JSON
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
