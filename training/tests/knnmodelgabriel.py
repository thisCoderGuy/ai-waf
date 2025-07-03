import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import io

# --- Configuration ---
LOG_FILE_PATHS = [
    os.path.join('csv', 'coraza-audit-benign.csv'),
    os.path.join('csv', 'coraza-audit-sqli.csv'),
    os.path.join('csv', 'coraza-audit-cleaned.csv'),
    os.path.join('csv', 'coraza-audit-xss.csv')
]
MODEL_OUTPUT_PATH = os.path.join('..', 'ai-microservice', 'knn_malicious_traffic_model.joblib')
PREPROCESSOR_OUTPUT_PATH = os.path.join('..', 'ai-microservice', 'knn_malicious_traffic_preprocessor.joblib')

COLUMNS_TO_DROP = ['Timestamp', 'TransactionID', 'ClientIP', 'ClientPort', 'ServerIP', 'ServerPort',
                   'ResponseProtocol', 'ResponseHeaders', 'ResponseBody', 'WAFInterrupted', 'InterruptionRuleID',
                   'InterruptionStatus', 'InterruptionAction', 'MatchedRulesCount', 'MatchedRulesIDs',
                   'MatchedRulesMessages', 'MatchedRulesTags', 'AIScore', 'AIVerdict']

problematic_endings = [
    "Failed to write CSV record: short write",
    "CSV writer error: short write"
]


# --- 1. Data Loading ---
def load_and_clean_data(log_file_paths, columns_to_drop=None):
    all_dfs = []
    for log_file in log_file_paths:
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                raw_lines = f.readlines()

            if not raw_lines:
                print(f"Warning: {log_file} is empty. Skipping.")
                continue

            header_line = raw_lines[0]
            data_lines = raw_lines[1:]
            filtered_data_lines = []

            for data_line in data_lines:
                stripped_line = data_line.rstrip('\r\n')
                keep_this_line = True
                for ending in problematic_endings:
                    if stripped_line.endswith(ending):
                        keep_this_line = False
                        break
                if keep_this_line:
                    filtered_data_lines.append(data_line)

            processed_content = [header_line] + filtered_data_lines
            if len(processed_content) <= 1:
                print(f"Warning: No valid CSV-like data lines found in {log_file}. Skipping.")
                continue

            data_io = io.StringIO("".join(processed_content))
            df = pd.read_csv(data_io, on_bad_lines='skip')
            all_dfs.append(df)
            print(f"Successfully loaded and pre-filtered {log_file}")

        except FileNotFoundError:
            print(f"Warning: Log file not found at {log_file}. Skipping.")
        except Exception as e:
            print(f"Error processing CSV file {log_file}: {e}. Skipping.")

    if not all_dfs:
        print("No data loaded from any specified files.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(df)} raw log entries from all files.")

    # --- Data Cleaning ---
    print("\nPerforming data cleaning...")
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_rows - len(df)} duplicate rows.")

    if 'AIVerdictLabel' in df.columns:
        df = df[df['AIVerdictLabel'].isin(['benign', 'malicious'])]
        print(f"Filtered rows by valid AIVerdictLabel.")

    critical_columns = ['RequestURI', 'RequestMethod']
    existing_critical_columns = [col for col in critical_columns if col in df.columns]
    if existing_critical_columns:
        df.dropna(subset=existing_critical_columns, inplace=True)
        print(f"Removed rows with missing critical features.")

    if 'RequestMethod' in df.columns:
        df['RequestMethod'] = df['RequestMethod'].astype(str).str.upper()

    if columns_to_drop:
        cols_to_drop_existing = [col for col in columns_to_drop if col in df.columns]
        df.drop(columns=cols_to_drop_existing, inplace=True)

    print(f"Remaining {len(df)} entries after cleaning.")
    return df


# --- 2. Preprocessing ---
def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('text_uri_path', TfidfVectorizer(max_features=5000, analyzer='char', ngram_range=(2, 4)), 'RequestURIPath'),
            ('text_uri_query', TfidfVectorizer(max_features=5000, analyzer='char', ngram_range=(2, 4)), 'RequestURIQuery'),
            ('text_body', TfidfVectorizer(max_features=5000, analyzer='char', ngram_range=(2, 4)), 'RequestBody'),
            ('text_user_agent', TfidfVectorizer(max_features=5000, analyzer='char', ngram_range=(2, 4)), 'UserAgent'),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['RequestMethod']),
            ('num', 'passthrough', ['RequestLength', 'PathLength', 'QueryLength'])
        ]
    )


def prepare_features(df):
    df['label'] = df['AIVerdictLabel'].apply(lambda x: 1 if x == 'malicious' else 0)
    df['RequestMethod'] = df.get('RequestMethod', 'UNKNOWN').fillna('')
    df['RequestURIPath'] = df.get('RequestURIPath', '').fillna('')
    df['RequestURIQuery'] = df.get('RequestURIQuery', '').fillna('')
    df['RequestBody'] = df.get('RequestBody', '').fillna('')
    df['UserAgent'] = df.get('UserAgent', '').fillna('')
    df['RequestLength'] = df.get('RequestLength', 0)
    df['PathLength'] = df.get('PathLength', 0)
    df['QueryLength'] = df.get('QueryLength', 0)
    return df


# --- 3. Main Execution ---
if __name__ == "__main__":
    df = load_and_clean_data(LOG_FILE_PATHS, columns_to_drop=COLUMNS_TO_DROP)
    if df.empty:
        print("No data to process. Exiting.")
        exit()

    df = prepare_features(df)
    y = df.pop('label')
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(df, y, test_size=0.2, stratify=y, random_state=42)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Train model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # --- Cross-Validation ---
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
    print("\n--- 5-Fold Cross-Validation on Training Set ---")
    print("CV Scores:", cv_scores)
    print("Average CV Score:", cv_scores.mean())

    # --- Evaluation ---
    y_pred = knn.predict(X_test)
    print("\n--- Test Set Evaluation ---")
    print(classification_report(y_test, y_pred, target_names=['benign', 'malicious']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and preprocessor
    joblib.dump(knn, MODEL_OUTPUT_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_OUTPUT_PATH)
    print("\nKNN model and preprocessor saved successfully.")

    #joblib