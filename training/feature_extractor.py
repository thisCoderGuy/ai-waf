import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df, tfidf_max_features, tfidf_analyzer, tfidf_ngram_range):
    """
    Extracts features and creates a ColumnTransformer for preprocessing.

    Args:
        df (pandas.DataFrame): The input DataFrame after initial cleaning.
        tfidf_max_features (int): Maximum number of features for TF-IDF vectorizers.
        tfidf_analyzer (str): Whether to use 'word' or 'char' for TF-IDF.
        tfidf_ngram_range (tuple): The n-gram range for character TF-IDF.

    Returns:
        tuple: A tuple containing:
            - preprocessor (ColumnTransformer): The fitted preprocessor.
            - X_processed (scipy.sparse.csr_matrix or numpy.ndarray): The transformed features.
            - y (pandas.Series): The target labels.
    """
    print("\nPerforming feature extraction and preprocessing...")

    # We use AIVerdictLabel as the label.
    df['label'] = df['AIVerdictLabel'].apply(lambda x: 1 if x == 'malicious' else 0)

    # Ensure required columns exist, or provide default empty strings/values
    # .fillna('') method ensures that any NaN (Not a Number) values in these text columns
    # are explicitly converted to empty strings ('').
    df['request_method'] = df.get('RequestMethod', 'UNKNOWN').fillna('')
    df['request_uri_path'] = df.get('RequestURIPath', '').fillna('')
    df['request_uri_query'] = df.get('RequestURIQuery', '').fillna('')
    df['request_body'] = df.get('RequestBody', '').fillna('')
    df['user_agent'] = df.get('UserAgent', '') # Using 'UserAgent' from original log

    # Numerical features (e.g., lengths)
    df['request_length'] = df.get('RequestLength', 0)
    df['path_length'] = df.get('PathLength', 0)
    df['query_length'] = df.get('QueryLength', 0)

    # Define preprocessing steps for different types of features
    text_features = [
        'request_uri_path',
        'request_uri_query',
        'request_body',
        'user_agent' # Add user_agent to text features
    ]
    categorical_features = ['request_method']
    numerical_features = ['request_length', 'path_length', 'query_length']

    # Create a ColumnTransformer to apply different transformers to different columns
    # When .fit() or .fit_transform() is called on this preprocessor object,
    # it will apply the specified transformers to their respective columns.
    # TfidfVectorizer: Term Frequency-Inverse Document Frequency, a numerical statistic
    # that reflects how important a word/character is to a document in a collection.
    # analyzer='char': Considers individual characters and sequences of characters (n-grams) as tokens.
    # This can be useful for detecting patterns in highly obfuscated attacks, misspellings,
    # or specific byte sequences that might not form meaningful words.
    # OneHotEncoder: learns all unique categorical values for each specified column (e.g., "GET", "POST").
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_uri_path', TfidfVectorizer(max_features=tfidf_max_features, analyzer=tfidf_analyzer, ngram_range=tfidf_ngram_range), 'request_uri_path'),
            ('text_uri_query', TfidfVectorizer(max_features=tfidf_max_features, analyzer=tfidf_analyzer, ngram_range=tfidf_ngram_range), 'request_uri_query'),
            ('text_body', TfidfVectorizer(max_features=tfidf_max_features, analyzer=tfidf_analyzer, ngram_range=tfidf_ngram_range), 'request_body'),
            ('text_user_agent', TfidfVectorizer(max_features=tfidf_max_features), 'user_agent'), # User-agent might benefit from word-level TF-IDF or char-level based on nature
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ],
        remainder='drop' # Drop columns not specified in transformers
    )

    # Fit the preprocessor on the data to learn vocabulary, categories, etc.
    X_processed = preprocessor.fit_transform(df)
    y = df['label']

    print("Feature extraction and preprocessing complete.")
    return preprocessor, X_processed, y
