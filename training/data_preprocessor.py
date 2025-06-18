import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

# Import model-specific configurations and tuning parameters
from config import (
    LABEL, TEXT_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    PERFORM_SPARSE_PREPROCESSING, TFIDF_MAX_FEATURES,
    TFIDF_ANALYZERS, TFIDF_NGRAM_RANGES,
    PERFORM_DENSE_PREPROCESSING, TOKENIZER_CONFIGS, 
    MAX_SEQ_LENGTHS
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
    
    # We use LABEL as the label
    df['label'] = df[LABEL].apply(lambda x: 1 if x == 'malicious' else 0)
    y = df['label']

   
    # Extract and fill missing values for numerical features
    for feature in NUMERICAL_FEATURES:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0)
        else:
            df[feature] = 0

    # Fill missing values for text and categorical features
    for feature in TEXT_FEATURES:
        if feature in df.columns:
            df[feature] = df[feature].fillna('')
        else:
            df[feature] = ''
    
    for feature in CATEGORICAL_FEATURES:
        if feature in df.columns:
            df[feature] = df[feature].fillna('')
        else:
            df[feature] = ''
    
    
    if PERFORM_SPARSE_PREPROCESSING:

        logger.info("--- Preprocessing (Sparse Feature extraction) ---")
        log_message = f"""\tTFIDF_MAX_FEATURES: {TFIDF_MAX_FEATURES}
    \tTFIDF_ANALYZERS: {TFIDF_ANALYZERS}
    \tTFIDF_NGRAM_RANGES: {TFIDF_NGRAM_RANGES}"""
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

        text_transformers = [
            (
                feature,
                TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES[feature],
                    analyzer=TFIDF_ANALYZERS[feature],
                    ngram_range=TFIDF_NGRAM_RANGES[feature]
                ),
                feature
            )
            for feature in TEXT_FEATURES
        ]

        full_transformers = text_transformers + [
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
            ('num', StandardScaler(), NUMERICAL_FEATURES)
        ]

        # Create the column transformer
        preprocessor = ColumnTransformer(
            transformers=full_transformers,
            remainder='drop'
        )



        # Fit the preprocessor on the data to learn vocabulary, categories, etc.
        X_processed = preprocessor.fit_transform(df)       

        print("Sparse feature extraction and preprocessing complete.")

    elif PERFORM_DENSE_PREPROCESSING:
        logger.info("--- Preprocessing (Dense Feature extraction) ---")
        log_message = f"""\tUsing character-level tokenization for text features.
    \tTOKENIZER_CONFIGS: {TOKENIZER_CONFIGS}
    \tMAX_SEQ_LENGTHS: {MAX_SEQ_LENGTHS}"""
        logger.info(log_message) 
        print("\nPerforming feature extraction and preprocessing...") 
        
        preprocessor = {}
        X_processed = {}

        # 1. Process Numerical Features
        for numerical_feature in NUMERICAL_FEATURES:
            print(f"Processing numerical feature: {numerical_feature}...")            
            num_scaler = StandardScaler()
            X_processed[numerical_feature] = num_scaler.fit_transform(df[numerical_feature]) 
            preprocessor[numerical_feature] = num_scaler
        # Save the number of numericalfeatures
        preprocessor["num_numerical_features"] = len(NUMERICAL_FEATURES)   
       

        # 2. Process Categorical Features with LabelEncoder
        for categorical_feature in CATEGORICAL_FEATURES:
            print(f"Processing categorical feature: {categorical_feature}...")            
            cat_encoder = LabelEncoder()
            encoded_labels = cat_encoder.fit_transform(df[categorical_feature])
            X_processed[categorical_feature] = encoded_labels.reshape(-1, 1)
            preprocessor[categorical_feature] = cat_encoder
            # Save the cardinality (number of unique values)
            preprocessor[f"{categorical_feature}_cardinality"] = len(cat_encoder.classes_)
        
        
        
        # 3. Process Text Features        
        for text_feature in TEXT_FEATURES:
            max_len = MAX_SEQ_LENGTHS[text_feature]
            is_char_tokenizer = TOKENIZER_CONFIGS[text_feature] == 'char'

            print(f"Processing text feature: {text_feature}...")
            # Initialize a character-level tokenizer
            # OOV token handles characters not seen during training
            char_tokenizer = Tokenizer(char_level=is_char_tokenizer, oov_token='<UNK>')
            
            # Fit the tokenizer on the text data to build the character vocabulary
            char_tokenizer.fit_on_texts(df[text_feature])
            
            # Convert text to sequences of integers 
            sequences = char_tokenizer.texts_to_sequences(df[text_feature])
            
            # Pad sequences to ensure they all have the same length
            padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
            
            # Store the processed data and the fitted tokenizer
            X_processed[text_feature] = padded_sequences
            preprocessor[text_feature] = char_tokenizer
            # store vocab_size for each text feature
            preprocessor[f"{text_feature}_vocab_size"] = len(char_tokenizer.word_index) + 1

        
        print("Dense feature extraction and preprocessing complete.")

    else:
        # No preprocessing
        preprocessor = None
        # Select the relevant columns as raw features if no preprocessing is done
        # The model consuming this X_processed would then be responsible for handling these raw features.
        X_processed = df[
            CATEGORICAL_FEATURES + TEXT_FEATURES + NUMERICAL_FEATURES
        ].copy()
    
    return preprocessor, X_processed, y
