import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin

from loggers import global_logger, evaluation_logger

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

# Import model-specific configurations and tuning parameters
from training_config import (
    LABEL, LABEL_VALUES, TEXT_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    TYPE_OF_PREPROCESSING, TFIDF_MAX_FEATURES,
    TFIDF_ANALYZERS, TFIDF_NGRAM_RANGES,
    TOKENIZER_CONFIGS, 
    MAX_SEQ_LENGTHS
)

def preprocess_data(df):
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
    
    evaluation_logger.info("--- Data Preprocessing ---")
    evaluation_logger.info(f"{NUMERICAL_FEATURES=}")
    evaluation_logger.info(f"{TEXT_FEATURES=}")
    evaluation_logger.info(f"{CATEGORICAL_FEATURES=}")
    evaluation_logger.info(f"{LABEL=}")
    # We use LABEL as the label
    df['label'] = df[LABEL].apply(lambda x: 1 if x == LABEL_VALUES[1] else 0)
    y = df['label']
    global_logger.debug(f"{type(y)=}")
    global_logger.debug(f"{y.shape=}")
    global_logger.debug(f"{y.name=}")
    global_logger.debug(f"{y.values[:3]=}")
    global_logger.debug(f"{y.values[-3:]=}")
   
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
    
    
    if TYPE_OF_PREPROCESSING == 'sparse':

        evaluation_logger.info("> Sparse Feature extraction")
        log_message = f"""\tTFIDF_MAX_FEATURES: {TFIDF_MAX_FEATURES}
    \tTFIDF_ANALYZERS: {TFIDF_ANALYZERS}
    \tTFIDF_NGRAM_RANGES: {TFIDF_NGRAM_RANGES}"""
        evaluation_logger.info(log_message) 

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

        evaluation_logger.info("Sparse feature extraction and preprocessing complete.")

    elif TYPE_OF_PREPROCESSING == 'dense':
        evaluation_logger.info("> Dense Feature extraction")
        log_message = f"""\tUsing :
    \tTOKENIZER_CONFIGS: {TOKENIZER_CONFIGS}
    \tMAX_SEQ_LENGTHS: {MAX_SEQ_LENGTHS}"""
        evaluation_logger.info(log_message) 
        
        preprocessor = DeepLearningMultiFeaturePreprocessor()
        preprocessor.fit(df)

        X_processed = preprocessor.transform(df)

        global_logger.debug(f"{X_processed.columns=}")
        evaluation_logger.info("Dense feature extraction and preprocessing complete.")

    else:
        # No preprocessing        
        evaluation_logger.info("> No Feature extraction")
        preprocessor = None
        # Select the relevant columns as raw features if no preprocessing is done
        # The model consuming this X_processed would then be responsible for handling these raw features.
        X_processed = df[
            CATEGORICAL_FEATURES + TEXT_FEATURES + NUMERICAL_FEATURES
        ].copy()

    
    return preprocessor, X_processed, y



class DeepLearningMultiFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    A custom preprocessor for combining numerical, categorical, and multiple text features.
    Uses StandardScaler for numerical, OneHotEncoder for categorical,
    and tf.keras.preprocessing.text.Tokenizer + pad_sequences for text.
    Designed to be saved/loaded with joblib.
    Exposes learned dimensions as properties for model instantiation.
    """
    def __init__(self):
        
        self.numerical_cols = NUMERICAL_FEATURES
        self.categorical_cols = CATEGORICAL_FEATURES
        self.text_cols = TEXT_FEATURES


        self.scaler = None
        self.onehot_encoder = None
        self.text_tokenizers = {}

        self._num_numerical_features_dim = 0
        self._num_categorical_features_dim = 0
        self._text_vocab_sizes = {} 
        self._combined_numerical_categorical_dim = 0
        self._num_text_input_columns = len(TEXT_FEATURES) 


    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # --- Fit Numerical Scaler ---
        if self.numerical_cols:
            global_logger.debug(f"Fitting StandardScaler for numerical features: {self.numerical_cols}...")  
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.numerical_cols].fillna(X[self.numerical_cols].mean()))
            self._num_numerical_features_dim = len(self.numerical_cols)
        else:
            self._num_numerical_features_dim = 0 

        # --- Fit OneHotEncoder for Categorical Columns ---
        if self.categorical_cols:
            global_logger.info(f"Fitting OneHotEncoder for categorical columns: {self.categorical_cols}")
            self.onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.onehot_encoder.fit(X[self.categorical_cols].astype(str).fillna('__missing__'))
            self._num_categorical_features_dim = self.onehot_encoder.get_feature_names_out(self.categorical_cols).shape[0]
        else:
            self._num_categorical_features_dim = 0 # Ensure it's 0 if no categorical columns

        # Calculate combined numerical/categorical dimension after fitting
        self._combined_numerical_categorical_dim = self._num_numerical_features_dim + self._num_categorical_features_dim


        # --- Fit Tokenizers for Text Columns ---
        for col in self.text_cols:
            is_char_tokenizer = TOKENIZER_CONFIGS[col] == 'char'

            global_logger.debug(f"Fitting Tokenizer on text feature: {col}...")
            # Initialize a character-level tokenizer
            # OOV token handles characters not seen during training
            char_tokenizer = Tokenizer(char_level=is_char_tokenizer, oov_token="<unk>", lower=True)   # Fit the tokenizer on the text data to build the character vocabulary
            texts = X[col].astype(str).fillna('')
            char_tokenizer.fit_on_texts(texts)
            self.text_tokenizers[col] = char_tokenizer
            
            self._text_vocab_sizes[col] = len(char_tokenizer.word_index) + 1
            global_logger.debug(f"  - Vocab size for '{col}': {self._text_vocab_sizes[col]}")
                 

        global_logger.debug("Preprocessor fitting complete.")
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_copy = X.copy()
        # Ensure all expected columns are present and filled with defaults if missing
        for col in self.numerical_cols:
            if col not in X_copy.columns: X_copy[col] = 0.0
            X_copy[col] = X_copy[col].fillna(0.0) # Fill NaNs with 0 for numerical
        for col in self.categorical_cols:
            if col not in X_copy.columns: X_copy[col] = '__missing__'
            X_copy[col] = X_copy[col].astype(str).fillna('__missing__')
        for col in self.text_cols:
            if col not in X_copy.columns: X_copy[col] = ''
            X_copy[col] = X_copy[col].astype(str).fillna('')

        # Initialize an empty Pandas DataFrame that has the same index as X_copy.
        X_processed = pd.DataFrame(index=X_copy.index)

        global_logger.debug("Preprocessor transforming starting.")
        # --- Transform Numerical Columns ---
        if self.numerical_cols:
            if self.scaler:
                global_logger.debug("Transforming numerical columns.")
                global_logger.debug(f"Before {X[self.numerical_cols][:3]=}")
                processed_data = self.scaler.transform(X_copy[self.numerical_cols])
                processed_df = pd.DataFrame(processed_data, index=X_copy.index, columns=[self.numerical_cols])
                global_logger.debug(f"After {processed_df[self.numerical_cols][:3]=}")
                X_processed = pd.concat([X_processed, processed_df], axis=1)
            else:
                X_processed = pd.concat([X_processed, X_copy[self.numerical_cols]], axis=1)


        # --- Transform Categorical Columns ---
        if self.categorical_cols:
            if self.onehot_encoder:
                global_logger.debug("Transforming categorical columns.")
                global_logger.debug(f"Before {X[self.categorical_cols][:3]=}")
                processed_categorical = self.onehot_encoder.transform(X_copy[self.categorical_cols])
                
                ohe_column_names = self.onehot_encoder.get_feature_names_out(self.categorical_cols)
        
                processed_df = pd.DataFrame(processed_categorical, index=X_copy.index, columns=ohe_column_names)
                global_logger.debug(f"After {processed_df[ohe_column_names][:3]=}")
                X_processed = pd.concat([X_processed, processed_df], axis=1)
            else:
                pass # Should not happen if categorical_cols is not empty and onehot_encoder is None
        
       

        
        # --- Transform Text Columns ---

        global_logger.debug("Transforming text columns.")
        for col in self.text_cols:
            if col in self.text_tokenizers:
                global_logger.debug(f"Processing text feature: {col}...")
                global_logger.debug(f"Before {X_copy[col][:3]=}") 
                tokenizer = self.text_tokenizers[col]
                # Convert text to sequences of integers 
                # takes a list of text strings and replaces each character (or word) in those strings with its corresponding integer ID from the tokenizer's vocabulary.
                sequences = tokenizer.texts_to_sequences(X_copy[col])


                max_len = MAX_SEQ_LENGTHS[col]               
                # Pad sequences to ensure they all have the same length
                padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
                
                global_logger.debug(f"{type(padded_sequences)=} {len(padded_sequences)=} {padded_sequences.shape=}")
                global_logger.debug(f"{padded_sequences[0]=}")
                global_logger.debug(f"{padded_sequences[1]=}")
                global_logger.debug(f"{padded_sequences[2]=}")
                
                # Store the processed data and the fitted tokenizer
                # Creates a new DataFrame from the padded_sequences array. 
                # Each column in this DataFrame will represent a position in the sequence, 
                # and the values will be the integer IDs.
                processed_df = pd.DataFrame(padded_sequences, index=X_copy.index )  # index=df.index: ensures that the newly created df retains the original row index from df.     
                # make the column names of the new DataFrame more descriptive
                processed_df = processed_df.add_prefix(f"{self.text_cols}_")   
                X_processed = pd.concat([X_processed, processed_df], axis=1) # axis=1: column-wise
                
            else:
                global_logger.warning(f"Tokenizer for text column '{col}' not found. Returning zeros for this feature.")
                final_output_tuple.append(np.zeros((len(X_copy), self.text_sequence_length), dtype=int))

        global_logger.debug("Preprocessor transform complete.")
        global_logger.debug(f"{X_processed.columns=}")

        return X_processed
    
    # --- Properties to expose learned dimensions ---
    @property
    def num_numerical_features_dim(self):
        """Returns the number of numerical features after preprocessing."""
        return self._num_numerical_features_dim

    @property
    def num_categorical_features_dim(self):
        """Returns the number of one-hot encoded categorical features."""
        return self._num_categorical_features_dim

    @property
    def combined_numerical_categorical_dim(self):
        """Returns the total dimension of combined numerical and one-hot encoded categorical features."""
        return self._combined_numerical_categorical_dim

    @property
    def text_vocab_sizes(self):
        """Returns a dictionary mapping text column names to their vocabulary sizes."""
        return self._text_vocab_sizes

    @property
    def text_sequence_length(self):
        """Returns the fixed length for padded text sequences."""
        return self.text_sequence_length # This is from init, not learned, but useful here

    @property
    def num_text_input_columns(self):
        """Returns the number of text columns being processed."""
        return self._num_text_input_columns
