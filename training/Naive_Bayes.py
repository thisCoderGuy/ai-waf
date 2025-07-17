import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. --- Config ---
DATA_PATHS = [
    ("training/training-data/cleaned/coraza-audit-cleaned.csv")
]

MODEL_PATH = "models/nb_model.joblib"
PREPROCESSOR_PATH = "models/nb_preprocessor.joblib"
LABEL_COLUMN = "AIVerdictLabel"

# 2. --- Load & Clean ---
def load_data(paths):
    dfs = []
    for path in paths:
        if os.path.exists(path):
            df = pd.read_csv(path, on_bad_lines="skip")
            dfs.append(df)
        else:
            print(f"Warning: {path} not found.")
    if not dfs:
        raise ValueError("No valid data files loaded.")
    data = pd.concat(dfs, ignore_index=True)
    data = data[data[LABEL_COLUMN].isin(["benign", "malicious"])]
    return data

# 3. --- Preprocessing ---
def build_preprocessor():
    text_features = ["RequestURIPath", "RequestURIQuery", "RequestBody", "UserAgent"]
    cat_features = ["RequestMethod","RequestProtocol"]
    num_features = ["RequestLength", "PathLength", "QueryLength", ]

    return ColumnTransformer([
        ("uri", TfidfVectorizer(analyzer='char', ngram_range=(2,4)), "RequestURIPath"),
        ("query", TfidfVectorizer(analyzer='char', ngram_range=(2,4)), "RequestURIQuery"),
        ("body", TfidfVectorizer(analyzer='char', ngram_range=(2,4)), "RequestBody"),
        ("agent", TfidfVectorizer(analyzer='char', ngram_range=(2,4)), "UserAgent"),
        ("method", OneHotEncoder(handle_unknown="ignore"), ["RequestMethod"]),
        ("protocol", OneHotEncoder(handle_unknown="ignore"), ["RequestProtocol"]),
        ("numeric", "passthrough", num_features)
    ])

# 4. --- Train Model ---
def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# 5. --- Evaluation ---
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=["benign", "malicious"]))

# 6. --- Save Artifacts ---
def save_artifacts(model, preprocessor):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print("Saved model and preprocessor.")

# 7. --- Main ---
def main():
    print("Loading data...")
    df = load_data(DATA_PATHS)

    # Fill missing values
    for col in ["RequestURIPath", "RequestURIQuery", "RequestBody", "UserAgent"]:
        df[col] = df.get(col, "").fillna("")
    df["RequestMethod"] = df.get("RequestMethod", "UNKNOWN").fillna("UNKNOWN")
    df["RequestLength"] = df.get("RequestLength", 0)
    df["PathLength"] = df.get("PathLength", 0)
    df["QueryLength"] = df.get("QueryLength", 0)

    y = df[LABEL_COLUMN].apply(lambda x: 1 if x == "malicious" else 0)
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    print("Training model...")
    model = train_naive_bayes(X_train, y_train)

    print("Evaluating model...")
    evaluate(model, X_test, y_test)

    print("Saving model and preprocessor...")
    save_artifacts(model, preprocessor)

if __name__ == "__main__":
    main()
