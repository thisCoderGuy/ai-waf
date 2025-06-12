from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from datetime import datetime # Import datetime for timestamps
import pandas as pd # Import pandas for better confusion matrix display

def evaluate_model(model, X_test, y_test, logger):
    """
    Evaluates the trained model and prints performance metrics,
    including the confusion matrix, to console and log file.

    Args:
        model (object): The trained machine learning model.
        X_test (array-like): Testing features.
        y_test (pandas.Series): Testing labels.
        logger (logging.Logger): Logger object to write evaluation details.
    """

    logger.info("--- Model Evaluation ---")
   

    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['benign', 'malicious'], output_dict=False) # Use output_dict=False for formatted string
    
    # Calculate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    # Create a DataFrame for better visualization of the confusion matrix
    cm_df = pd.DataFrame(cm, index=['Actual Benign', 'Actual Malicious'], columns=['Predicted Benign', 'Predicted Malicious'])

    # Format the evaluation details for logging
    log_message = f"""\t--- Evaluation Results ---
\tAccuracy: {accuracy:.4f}
\tPrecision: {precision:.4f}
\tRecall: {recall:.4f}
\tF1-Score: {f1:.4f}

\t--- Classification Report ---
{report}

\t--- Confusion Matrix ---
{cm_df.to_string()}
"""
    # Print to console
    print(log_message)

    # Log to file
    logger.info(log_message)
    print("Evaluation results logged.")
