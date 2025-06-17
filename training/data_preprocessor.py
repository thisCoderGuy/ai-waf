import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

# Import model-specific configurations and tuning parameters
from config import (
    PERFORM_SPARSE_PREPROCESSING, TFIDF_MAX_FEATURES,
    URI_PATH_TFIDF_ANALYZER, URI_PATH_TFIDF_NGRAM_RANGE,
    URI_QUERY_TFIDF_ANALYZER, URI_QUERY_TFIDF_NGRAM_RANGE,
    BODY_TFIDF_ANALYZER, BODY_TFIDF_NGRAM_RANGE, 
    USER_AGENT_TFIDF_ANALYZER, USER_AGENT_TFIDF_NGRAM_RANGE,
    PERFORM_DENSE_PREPROCESSING, MAX_SEQ_LENGTH_PATH, MAX_SEQ_LENGTH_QUERY,
    MAX_SEQ_LENGTH_BODY, MAX_SEQ_LENGTH_USER_AGENT
)

def preprocess_data(df, logger):
    """
    Extracts features and creates a ColumnTransformer for preprocessing.

    Args:
        df (pandas.DataFrame): The input DataFrame after initial cleaning.
        logger: A logger object for logging messages.

    Returns:
        tuple: A tuple containing:
          - preprocessor (ColumnTransformer or None): The fitted preprocessor if preprocessing is performed, else None.
          - X_processed (scipy.sparse.csr_matrix, numpy.ndarray, or pandas.DataFrame): The transformed features
                                                                                         or raw features.
          - y (pandas.Series): The target labels.
    """
    
    # We use AIVerdictLabel as the label.
    df['label'] = df['AIVerdictLabel'].apply(lambda x: 1 if x == 'malicious' else 0)
    y = df['label']


    # Define preprocessing steps for different types of features
    text_features = [
            'request_uri_path',
            'request_uri_query',
            'request_body',
            'user_agent' 
    ]
    categorical_features = ['request_method']
    numerical_features = ['request_length', 'path_length', 'query_length']

    # Extract and fill missing values for numerical features
    df['request_length'] = df.get('RequestLength', 0).fillna(0)
    df['path_length'] = df.get('PathLength', 0).fillna(0)
    df['query_length'] = df.get('QueryLength', 0).fillna(0)

    # Fill missing values for text and categorical features
    df['request_uri_path'] = df.get('RequestURIPath', '').fillna('')
    df['request_uri_query'] = df.get('RequestURIQuery', '').fillna('')
    df['request_body'] = df.get('RequestBody', '').fillna('')
    df['user_agent'] = df.get('UserAgent', '').fillna('') 
    
    df['request_method'] = df.get('RequestMethod', 'UNKNOWN').fillna('')


    if PERFORM_SPARSE_PREPROCESSING:

        logger.info("--- Preprocessing (Sparse Feature extraction) ---")
        log_message = f"""\tTFIDF_MAX_FEATURES: {TFIDF_MAX_FEATURES}
    \tURI_PATH_TFIDF_ANALYZER: {URI_PATH_TFIDF_ANALYZER}
    \tURI_PATH_TFIDF_NGRAM_RANGE: {URI_PATH_TFIDF_NGRAM_RANGE}
    \tURI_QUERY_TFIDF_ANALYZER: {URI_QUERY_TFIDF_ANALYZER}
    \tURI_QUERY_TFIDF_NGRAM_RANGE: {URI_QUERY_TFIDF_NGRAM_RANGE}
    \tBODY_TFIDF_ANALYZER: {BODY_TFIDF_ANALYZER}
    \tBODY_TFIDF_NGRAM_RANGE: {BODY_TFIDF_NGRAM_RANGE}
    \tUSER_AGENT_TFIDF_ANALYZER: {USER_AGENT_TFIDF_ANALYZER}
    \tUSER_AGENT_TFIDF_NGRAM_RANGE: {USER_AGENT_TFIDF_NGRAM_RANGE}"""
        logger.info(log_message) 
        print("\nPerforming feature extraction and preprocessing...")  

        # Create a ColumnTransformer to apply different transformers to different columns
        # When .fit() or .fit_transform() is called on this preprocessor object,
        # it will apply the specified transformers to their respective columns.
        # TfidfVectorizer: Term Frequency-Inverse Document Frequency, a numerical statistic
        # that reflects how important a word/character is to a document in a collection.
        # analyzer='char': Considers individual characters and sequences of characters (n-grams) as tokens.
        # This can be useful for detecting patterns in highly obfuscated attacks, misspellings,
        # or specific byte sequences that might not form meaningful words.
        # OneHotEncoder: learns all unique categorical values for each specified column (e.g., "GET", "POST").
        # StandardScaler: Normalizes numerical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('text_uri_path', TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, analyzer=URI_PATH_TFIDF_ANALYZER, ngram_range=URI_PATH_TFIDF_NGRAM_RANGE), 'request_uri_path'),
                ('text_uri_query', TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, analyzer=URI_QUERY_TFIDF_ANALYZER, ngram_range=URI_QUERY_TFIDF_NGRAM_RANGE), 'request_uri_query'),
                ('text_body', TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, analyzer=BODY_TFIDF_ANALYZER, ngram_range=BODY_TFIDF_NGRAM_RANGE), 'request_body'),
                ('text_user_agent', TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, analyzer=USER_AGENT_TFIDF_ANALYZER, ngram_range=USER_AGENT_TFIDF_NGRAM_RANGE), 'user_agent'), 
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                ('num', StandardScaler(), numerical_features)
            ],
            remainder='drop' # Drop columns not specified in transformers
        )
        # Fit the preprocessor on the data to learn vocabulary, categories, etc.
        X_processed = preprocessor.fit_transform(df)       

        print("Sparse feature extraction and preprocessing complete.")

    elif PERFORM_DENSE_PREPROCESSING:
        logger.info("--- Preprocessing (Dense Feature extraction) ---")
        log_message = f"""\tUsing character-level tokenization for text features.
    \tMAX_SEQ_LENGTH_PATH: {MAX_SEQ_LENGTH_PATH}
    \tMAX_SEQ_LENGTH_QUERY: {MAX_SEQ_LENGTH_QUERY}
    \tMAX_SEQ_LENGTH_BODY: {MAX_SEQ_LENGTH_BODY}
    \tMAX_SEQ_LENGTH_USER_AGENT: {MAX_SEQ_LENGTH_USER_AGENT}"""
        logger.info(log_message) 
        print("\nPerforming feature extraction and preprocessing...") 
        
        preprocessor = {}
        X_processed = {}

        # 1. Process Numerical Features
        print("Processing numerical features...")
        num_scaler = StandardScaler()
        X_processed['numerical'] = num_scaler.fit_transform(df[numerical_features]) #The numerical_features list contains three column names: ['request_length', 'path_length', 'query_length'].
        preprocessor['numerical'] = num_scaler

       # 2. Process Categorical Features with LabelEncoder
        print("Processing categorical features...")
        cat_encoder = LabelEncoder()
        # LabelEncoder expects a 1D array, so we select the first column.
        # This is safe as categorical_features only contains 'request_method'.
        encoded_labels = cat_encoder.fit_transform(df[categorical_features].iloc[:, 0])
        # Reshape to be a column vector (e.g., (n_samples, 1)) for the model input
        X_processed['categorical'] = encoded_labels.reshape(-1, 1)
        preprocessor['categorical'] = cat_encoder

        # 3. Process Text Features
        # A dictionary to map feature names to their max sequence lengths from config
        text_feature_configs = {
            'request_uri_path': MAX_SEQ_LENGTH_PATH,
            'request_uri_query': MAX_SEQ_LENGTH_QUERY,
            'request_body': MAX_SEQ_LENGTH_BODY,
            'user_agent': MAX_SEQ_LENGTH_USER_AGENT
        }

        for text_feature, max_len in text_feature_configs.items():
            print(f"Processing text feature: {text_feature}...")
            # Initialize a character-level tokenizer
            # OOV token handles characters not seen during training
            char_tokenizer = Tokenizer(char_level=True, oov_token='<UNK>')
            
            # Fit the tokenizer on the text data to build the character vocabulary
            char_tokenizer.fit_on_texts(df[text_feature])
            
            # Convert text to sequences of integers
            sequences = char_tokenizer.texts_to_sequences(df[text_feature])
            
            # Pad sequences to ensure they all have the same length
            padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
            
            # Store the processed data and the fitted tokenizer
            X_processed[text_feature] = padded_sequences
            preprocessor[text_feature] = char_tokenizer
        
        print("Dense feature extraction and preprocessing complete.")

    else:
        # No preprocessing
        preprocessor = None
        # Select the relevant columns as raw features if no preprocessing is done
        # The model consuming this X_processed would then be responsible for handling these raw features.
        X_processed = df[
            ['request_method', 'request_uri_path', 'request_uri_query', 
             'request_body', 'user_agent', 'request_length', 'path_length', 'query_length']
        ].copy()
    
    return preprocessor, X_processed, y
