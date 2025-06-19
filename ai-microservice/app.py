from flask import Flask, request, jsonify
import logging
import joblib
import pandas as pd
import os

from config import (
    MODEL_PATH, PREPROCESSOR_PATH 
)

app = Flask(__name__)

# Configure logging to stdout/stderr which Docker captures
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Global variables to hold the loaded model and preprocessor
model = None
preprocessor = None

def load_ml_assets():
    """Loads the trained model and preprocessor on application startup."""
    global model, preprocessor
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logging.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            logging.error(f"Model file not found at {MODEL_PATH}")
            model = None

        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logging.info(f"Preprocessor loaded successfully from {PREPROCESSOR_PATH}")
        else:
            logging.error(f"Preprocessor file not found at {PREPROCESSOR_PATH}")
            preprocessor = None

    except Exception as e:
        logging.error(f"Error loading ML assets: {e}")
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
