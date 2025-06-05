import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier # Changed from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib # Preferred for scikit-learn models
import os # Import os for file path checks
import io # Import io for StringIO

# --- Configuration ---
# Path to Coraza audit log files
LOG_FILE_PATHS = [
    os.path.join('training_data', 'coraza-audit-benign.csv'),
    os.path.join('training_data', 'coraza-audit-sqli.csv'),
    os.path.join('training_data', 'coraza-audit-xss.csv')
]
# Path to save the trained model (moved to ai_microservice folder, one level up)
# Changed model name to reflect MLP
MODEL_OUTPUT_PATH = os.path.join('..', 'ai_microservice', 'mlp_malicious_traffic_model.joblib')
PREPROCESSOR_OUTPUT_PATH = os.path.join('..', 'ai_microservice', 'mlp_malicious_traffic_preprocessor.joblib')


# Columns to drop during data cleaning
COLUMNS_TO_DROP = ['Timestamp', 'TransactionID', 'ClientIP', 'ClientPort', 'ServerIP', 'ServerPort',
                   'ResponseProtocol', 'ResponseHeaders', 'ResponseBody',
                   'WAFInterrupted', 'InterruptionRuleID', 'InterruptionStatus', 'InterruptionAction',
                   'MatchedRulesCount', 'MatchedRulesIDs', 'MatchedRulesMessages', 'MatchedRulesTags',
                   'AIScore', 'AIVerdict' ] # 'TimeOfDayHour', 'TimeOfDayDayOfWeek'

# Path to save the cleaned dataset
CLEANED_DATA_OUTPUT_PATH = os.path.join('training_data', 'coraza-audit-cleaned.csv')


# --- 1. Data Loading ---
def load_and_clean_data(log_file_paths, columns_to_drop=None):
    """
    Loads and merges CSV log entries from the specified list of files,
    then performs initial data cleaning.
    """
    all_dfs = []
    for log_file in log_file_paths:
        try:
            # Read file content as raw lines first to pre-filter
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                raw_lines = f.readlines()

            if not raw_lines:
                print(f"Warning: {log_file} is empty. Skipping.")
                continue

            # Identify the header line (usually the first line)
            header_line = raw_lines[0]
            data_lines = raw_lines[1:]

            # Filter data lines: keep only those that contain at least one comma
            # This helps remove lines that are clearly not structured CSV rows
            # (e.g., plain log messages, incomplete lines without delimiters).
            # Note: This is a heuristic. Valid single-column CSVs or CSVs with
            # quoted commas will be handled by pandas.read_csv.
            filtered_data_lines = [line for line in data_lines if ',' in line]

            # Reconstruct the content for pandas, ensuring the header is always included
            processed_content = [header_line] + filtered_data_lines

            if len(processed_content) <= 1: # Only header or no data lines
                print(f"Warning: No valid CSV-like data lines found in {log_file} after initial filtering. Skipping.")
                continue

            # Use io.StringIO to treat the filtered lines as a file-like object for pandas
            data_io = io.StringIO("".join(processed_content))

            # Use on_bad_lines='skip' for any remaining parsing errors within the filtered lines
            df = pd.read_csv(data_io, on_bad_lines='skip')
            all_dfs.append(df)
            print(f"Successfully loaded and pre-filtered {log_file}")
            print(f"  (Original lines: {len(raw_lines)}, Filtered lines for CSV parsing: {len(processed_content)})")

        except FileNotFoundError:
            print(f"Warning: Log file not found at {log_file}. Skipping.")
        except Exception as e:
            # This catch will now primarily handle errors other than tokenizing data
            print(f"Error processing CSV file {log_file}: {e}. Skipping.")

    if not all_dfs:
        print("No data loaded from any specified files.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(df)} raw log entries from all files.")

    # --- Data Cleaning Steps ---
    print("\nPerforming data cleaning...")
    initial_rows = len(df)

    # 1. Remove rows where the first field contains "Failed" or "Error"
    if not df.empty and len(df.columns) > 0:
        first_column_name = df.columns[0]
        # Ensure the column is treated as string to use .str.contains
        df_filtered = df[~df[first_column_name].astype(str).str.contains("Failed|Error", case=False, na=False)]
        rows_removed_first_field = initial_rows - len(df_filtered)
        if rows_removed_first_field > 0:
            print(f"Removed {rows_removed_first_field} rows where the first field ('{first_column_name}') contained 'Failed' or 'Error'.")
            df = df_filtered
        else:
            print(f"No rows removed based on 'Failed'/'Error' in the first field ('{first_column_name}').")
    else:
        print("Warning: DataFrame is empty or has no columns. Skipping 'Failed'/'Error' row removal.")
    initial_rows = len(df) # Update initial_rows for next step


    # 2. Remove exact duplicate rows
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_rows - len(df)} duplicate rows.")
    initial_rows = len(df) # Update initial_rows for next step

    # 3. Remove rows where 'AIVerdictLabel' is missing or not 'benign'/'malicious'
    if 'AIVerdictLabel' in df.columns:
        df = df[df['AIVerdictLabel'].isin(['benign', 'malicious'])]
        print(f"Removed {initial_rows - len(df)} rows with invalid AIVerdictLabel.")
    else:
        print("Warning: 'AIVerdictLabel' column not found for cleaning. Skipping label-based row removal.")
    initial_rows = len(df)

    # 4. Handle missing values in critical feature columns
    critical_columns = ['RequestURI', 'RequestMethod']
    existing_critical_columns = [col for col in critical_columns if col in df.columns]

    if existing_critical_columns:
        df.dropna(subset=existing_critical_columns, inplace=True)
        print(f"Removed {initial_rows - len(df)} rows with missing critical features.")
    else:
        print("Warning: No critical feature columns found for missing value cleaning.")

    # 5. Standardize 'RequestMethod' to uppercase
    if 'RequestMethod' in df.columns:
        df['RequestMethod'] = df['RequestMethod'].astype(str).str.upper()
        print("Standardized 'RequestMethod' to uppercase.")
    else:
        print("Warning: 'RequestMethod' column not found for standardization.")

    # 6. Remove specific columns if provided
    if columns_to_drop:
        cols_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
        if cols_to_drop_existing:
            df.drop(columns=cols_to_drop_existing, inplace=True)
            print(f"Removed columns: {', '.join(cols_to_drop_existing)}")
        else:
            print("No specified columns to drop were found in the DataFrame.")


    print(f"Remaining {len(df)} entries after cleaning.")
    return df

# --- 2. Feature Extraction & Preprocessing ---
def preprocess_data(df):
    """
    Extracts features and creates a ColumnTransformer for preprocessing.
    Returns the preprocessor and the transformed features.
    """
    # We use AIVerdictLabel as the label.
    df['label'] = df['AIVerdictLabel'].apply(lambda x: 1 if x == 'malicious' else 0)

    # Ensure required columns exist, or provide default empty strings/values
    # .fillna('') method  ensures that any NaN (Not a Number) values in these text columns are explicitly converted to empty strings ('')
    df['request_method'] = df.get('RequestMethod', 'UNKNOWN').fillna('')
    df['request_uri_path'] = df.get('RequestURIPath', '').fillna('')
    df['request_uri_query'] = df.get('RequestURIQuery', '').fillna('')
    df['request_body'] = df.get('RequestBody', '').fillna('')
    df['user_agent'] = df.get('UserAgent', '').fillna('') # Ensure user_agent is handled

    # Numerical features (e.g., lengths)
    df['request_length'] = df.get('RequestLength', 0)
    df['path_length'] = df.get('PathLength', 0)
    df['query_length'] = df.get('QueryLength', 0)

    # Purpose of Preprocessing: To convert raw, heterogeneous data into a consistent, numerical format that machine learning algorithms can understand and process.
    # Define preprocessing steps for different types of features
    text_features = ['request_uri_path', 'request_uri_query', 'request_body', 'user_agent'] # Corrected 'agent' to 'user_agent'
    categorical_features = ['request_method']
    numerical_features = ['request_length', 'path_length', 'query_length']

    # Create a ColumnTransformer to apply different transformers to different columns
    # When .fit() or .fit_transform() is called on this preprocessor object, it will apply the specified transformers to their respective columns.
    # TfidfVectorizer(max_features=5000): TF-IDF: Term Frequency-Inverse Document Frequency,  a numerical statistic that reflects how important a word is to a document in a collection or corpus
    #     analyzer='char': The vectorizer will now consider individual characters and sequences of characters (n-grams) as tokens, rather than whole words. We hope this will be useful for detecting patterns in highly obfuscated attacks, misspellings, or specific byte sequences that might not form meaningful words.
    # OneHotEncoder: learns all unique categorical values for each specified column (e.g., "GET", "POST" for RequestMethod).
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_uri_path', TfidfVectorizer(max_features=5000, analyzer='char', ngram_range=(2, 4)), 'request_uri_path'),
            ('text_uri_query', TfidfVectorizer(max_features=5000, analyzer='char', ngram_range=(2, 4)), 'request_uri_query'),
            ('text_body', TfidfVectorizer(max_features=5000, analyzer='char', ngram_range=(2, 4)), 'request_body'),
            ('text_user_agent', TfidfVectorizer(max_features=5000, analyzer='char', ngram_range=(2, 4)), 'user_agent'), # Changed to char-level as well
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ],
        remainder='drop'
    )

    # Fit the preprocessor on the data to learn vocabulary, categories, etc.
    X_processed = preprocessor.fit_transform(df)

    return preprocessor, X_processed, df['label']

# --- 3. Model Training (MLPClassifier) ---
def train_model(X_train, y_train):
    """Trains an MLPClassifier model."""
    print("Training MLP model...")
    # Initializing MLPClassifier with some common parameters
    # hidden_layer_sizes: tuple, i-th element represents the number of neurons in the i-th hidden layer.
    # max_iter: Maximum number of iterations for the solver to converge.
    # random_state: For reproducibility.
    # activation: 'relu' is common.
    # solver: 'adam' is a good default for larger datasets.
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, random_state=42, activation='relu', solver='adam')
    model.fit(X_train, y_train)
    print("MLP model training complete.")
    return model

# --- 4. Model Evaluation ---
def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and prints performance metrics."""
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['benign', 'malicious']))

# --- 5. Model Export ---
def save_model_and_preprocessor(model, preprocessor, model_path, preprocessor_path):
    """Saves the trained model and preprocessor to specified paths."""
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

    print(f"\nSaving trained model to {model_path}...")
    joblib.dump(model, model_path)
    print(f"Saving preprocessor to {preprocessor_path}...")
    joblib.dump(preprocessor, preprocessor_path)
    print("Model and preprocessor saved successfully.")


# --- Main Orchestration Function ---
def main():
    # Create output directories if they don't exist
    os.makedirs('configs', exist_ok=True)
    os.makedirs('../predictive_model', exist_ok=True)

    

    # 1. Load and Clean Data
    df = load_and_clean_data(LOG_FILE_PATHS, columns_to_drop=COLUMNS_TO_DROP)

    if df.empty:
        print("No data remaining after cleaning. Cannot proceed with training.")
        return

    # Save the cleaned data to a new CSV file
    try:
        df.to_csv(CLEANED_DATA_OUTPUT_PATH, index=False)
        print(f"\nCleaned data saved to {CLEANED_DATA_OUTPUT_PATH}")
    except Exception as e:
        print(f"Error saving cleaned data to CSV: {e}")

    # 2. Preprocess Data
    preprocessor, X, y = preprocess_data(df)

    # Ensure there are enough samples and features after preprocessing
    if X.shape[0] == 0 or X.shape[1] == 0:
        print("Error: No features generated after preprocessing. Check your data and preprocessing steps.")
        return
    if len(y.unique()) < 2:
        print("Warning: Only one class present in labels. Cannot perform classification.")
        print("Ensure your dummy data or real data has both 'benign' and 'malicious' examples.")
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # 3. Train Model
    model = train_model(X_train, y_train)

    # 4. Evaluate Model
    evaluate_model(model, X_test, y_test)

    # 5. Save Model and Preprocessor
    save_model_and_preprocessor(model, preprocessor, MODEL_OUTPUT_PATH, PREPROCESSOR_OUTPUT_PATH)

    print("\nTo reuse the model in your AI microservice, you will need to:")
    print(f"1. Load the preprocessor: `preprocessor = joblib.load('{PREPROCESSOR_OUTPUT_PATH}')`")
    print(f"2. Load the model: `model = joblib.load('{MODEL_OUTPUT_PATH}')`")
    print("3. For new data, extract raw features, then transform them using the loaded preprocessor:")
    print("   `new_data_df = pd.DataFrame([{'RequestMethod': 'GET', 'RequestURIPath': '/some/path', 'RequestURIQuery': 'param=value', 'RequestBody': '', 'UserAgent': 'Mozilla/5.0', 'RequestLength': 100, 'PathLength': 10, 'QueryLength': 10}])`")
    print("4. Make predictions: `prediction = model.predict(new_features_processed)`")
    print("   `prediction_proba = model.predict_proba(new_features_processed)` (if probability=True was set)")


if __name__ == "__main__":
    main()
