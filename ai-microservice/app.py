from flask import Flask, request, jsonify
import logging
import joblib
import pandas as pd
import os

from config import (
     MODEL_PATH, PREPROCESSOR_PATH, # Existing explicit paths (can be used as fallback/initial defaults)
    BASE_MODEL_DIR, LATEST_MODEL_INFO_PATH # Paths for dynamic loading
)

app = Flask(__name__)

# Configure logging to stdout/stderr which Docker captures
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Global variables to hold the loaded model and preprocessor
model = None
preprocessor = None

def load_ml_assets():
    """
    Loads the trained model and preprocessor on application startup.
    Prioritizes loading the latest model info from LATEST_MODEL_INFO_PATH.
    Falls back to explicitly defined MODEL_PATH/PREPROCESSOR_PATH if latest info is not found.
    """
    global model, preprocessor
   
    current_model_path = None
    current_preprocessor_path = None
    
    logging.info("Attempting to load ML assets...")

    try:
        # --- Attempt to load from LATEST_MODEL_INFO_PATH first ---
        if os.path.exists(LATEST_MODEL_INFO_PATH):
            logging.info(f"Reading latest model info from {LATEST_MODEL_INFO_PATH}")
            with open(LATEST_MODEL_INFO_PATH, 'r') as f:
                model_filename_from_info = None
                preprocessor_filename_from_info = None
                for line in f:
                    if line.startswith("model_filename="):
                        model_filename_from_info = line.split('=', 1)[1].strip()
                    elif line.startswith("preprocessor_filename="):
                        preprocessor_filename_from_info = line.split('=', 1)[1].strip()
            
            if model_filename_from_info:
                current_model_path = os.path.join(BASE_MODEL_DIR, model_filename_from_info)
                current_preprocessor_path = os.path.join(BASE_MODEL_DIR, preprocessor_filename_from_info) if preprocessor_filename_from_info else None
                logging.info(f"Determined latest model: {current_model_path}")
            else:
                logging.warning(f"'{LATEST_MODEL_INFO_PATH}' found but no valid 'model_filename' entry. Falling back to explicit paths.")
        else:
            logging.warning(f"'{LATEST_MODEL_INFO_PATH}' not found. Falling back to explicit paths from config.")

        # --- Fallback to explicit paths if latest info couldn't be used ---
        if current_model_path is None:
            current_model_path = MODEL_PATH
            current_preprocessor_path = PREPROCESSOR_PATH
            logging.info(f"Using explicit model paths: Model={current_model_path}, Preprocessor={current_preprocessor_path}")

        # --- Load the model and preprocessor using the determined paths ---
        if current_model_path and os.path.exists(current_model_path):
            # Using the unified load_model_and_preprocessor from model_utils
            model = joblib.load(current_model_path)            
            preprocessor = joblib.load(current_preprocessor_path)
            
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
