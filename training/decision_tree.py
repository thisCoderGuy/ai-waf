import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# --- Configuration ---
LOG_FILE_PATHS = [
    os.path.join('training_data', 'raw', 'coraza-audit-benign.csv'),
    os.path.join('training_data', 'raw', 'coraza-audit-sqli.csv'),
    os.path.join('training_data', 'raw', 'coraza-audit-xss.csv')
]
MODEL_OUTPUT_PATH = os.path.join('..', 'ai-microservice')
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

# Timestamped filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_FILENAME = f"decision_tree_model_{timestamp}.joblib"

# --- Load and Clean Data ---
def load_data(paths):
    dfs = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"Loaded {len(df)} rows from {path}")
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    return df

# --- Preprocessing ---
def preprocess(df):
    df = df[df['AIVerdictLabel'].isin(['benign', 'malicious'])]  # Ensure only known labels
    df = df.dropna(subset=['AIVerdictLabel'])

    df['label'] = df['AIVerdictLabel'].apply(lambda x: 1 if x == 'malicious' else 0)

    X = df.drop(columns=['AIVerdictLabel', 'AIVulnerabilityTypeLabel'], errors='ignore')
    X = X.select_dtypes(include='number')
    y = df['label']
    return X, y

# --- Training with Stratified K-Fold ---
def train_model(X, y, n_splits=5):
    print(f"\nPerforming {n_splits}-fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accs, precs, recs, f1s = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        accs.append(accuracy_score(y_val, preds))
        precs.append(precision_score(y_val, preds))
        recs.append(recall_score(y_val, preds))
        f1s.append(f1_score(y_val, preds))

        print(f"\n--- Fold {fold + 1} ---")
        print(f"Accuracy: {accs[-1]:.4f}")
        print(f"Precision: {precs[-1]:.4f}")
        print(f"Recall: {recs[-1]:.4f}")
        print(f"F1 Score: {f1s[-1]:.4f}")

    print("\n--- Average Metrics ---")
    print(f"Accuracy: {np.mean(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"F1 Score: {np.mean(f1s):.4f}")

    # Final training on all data
    final_model = DecisionTreeClassifier(random_state=42)
    final_model.fit(X, y)
    return final_model

# --- Save Model ---
def save_model(model, filename):
    output_path = os.path.join(MODEL_OUTPUT_PATH, filename)
    joblib.dump(model, output_path)
    print(f"\n✅ Model saved to: {output_path}")

# --- Main ---
def main():
    df = load_data(LOG_FILE_PATHS)
    if df.empty:
        print("Dataset is empty. Aborting.")
        return

    X, y = preprocess(df)

    if X.empty or y.nunique() < 2:
        print("Error: Not enough data or label variety to train a model.")
        return

    model = train_model(X, y)
    save_model(model, MODEL_FILENAME)

    # Final classification report
    y_pred = model.predict(X)
    print("\n--- Final Classification Report on Full Data ---")
    print(classification_report(y, y_pred, target_names=['benign', 'malicious']))

if __name__ == "__main__":
    main()
