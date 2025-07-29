import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import numpy as np

from training_config import (
    TEXT_FEATURES, CATEGORICAL_FEATURES,  NUMERICAL_FEATURES
)
from loggers import global_logger

class HTTPRequestMultiInputDataset(Dataset):
    def __init__(self, X, y):
        """
        Initializes a multi input dataset from preprocessed features.

        Args:
            X (pd.DataFrame): Preprocessed training features (Pandas DataFrame).
            y (pd.Series): Training labels (Pandas Series).
        """
        
        global_logger.debug("Building Multi Input Dataset")
        global_logger.debug(f"from X.columns: {X.columns.tolist()=}") # Use .tolist() for better logging of long lists

        self.text_features = OrderedDict()
        
        # 1. Process Text Features
        global_logger.debug(" - Processing text columns to PyTorch tensors")
        for text_feature_name in TEXT_FEATURES:
            # Collect all columns that belong to this specific text_feature (e.g., 'RequestURIPath_0', 'RequestURIPath_1', ...)
            prefix = f"{text_feature_name}_"
            # Ensure columns are sorted correctly by their numeric suffix
            text_columns_for_this_feature = sorted([
                col for col in X.columns if col.startswith(prefix) and col[len(prefix):].isdigit()
            ], key=lambda x: int(x[len(prefix):]))
            
            if not text_columns_for_this_feature:
                global_logger.warning(f"No columns found for text feature '{text_feature_name}'. Skipping.")
                continue # Skip if no columns found for this feature

            # Convert to torch.long as these are token IDs for embedding
            feature_tensor = torch.tensor(X[text_columns_for_this_feature].values, dtype=torch.long)            
            self.text_features[text_feature_name] = feature_tensor
            global_logger.debug(f"      {text_feature_name=}: {feature_tensor.shape=} {feature_tensor.dtype=} {feature_tensor.ndim=}")
            
        # 2. Process Categorical Features (Combined into a single tensor)
        global_logger.debug(" - Processing categorical columns to a single PyTorch tensor")
        all_categorical_cols = []
        for cat_feature_name in CATEGORICAL_FEATURES:
            prefix = f"{cat_feature_name}_"
            # Collect all one-hot encoded columns for this categorical feature
            # Assuming OneHotEncoder creates names like 'RequestMethod_GET', 'RequestMethod_POST'
            categorical_ohe_columns = [col for col in X.columns if col.startswith(prefix)]
            all_categorical_cols.extend(categorical_ohe_columns)
        
        if all_categorical_cols:
            # Convert to torch.float32 as these are one-hot encoded (binary) values
            self.all_categorical_features = torch.tensor(X[all_categorical_cols].values, dtype=torch.float32)
            global_logger.debug(f"      All Categorical: {self.all_categorical_features.shape=} {self.all_categorical_features.dtype=} {self.all_categorical_features.ndim=}")
        else:
            self.all_categorical_features = torch.empty((len(X), 0), dtype=torch.float32) # Empty tensor if no categorical
            global_logger.debug(f"      No categorical features found. Empty tensor created: {self.all_categorical_features.shape=}")


        # 3. Process Numerical Features (Combined into a single tensor)
        global_logger.debug("- Processing numerical columns to a single PyTorch tensor")
        if NUMERICAL_FEATURES:
            # Ensure columns are selected correctly, handling potential tuple names from ColumnTransformer if not flattened
            numerical_cols_in_X = [col for col in NUMERICAL_FEATURES if col in X.columns or (isinstance(col, tuple) and col[0] in X.columns)]
            # If your preprocessor correctly flattens names, the `col in X.columns` check is enough.
            # If it returns tuples, the `(isinstance(col, tuple) and col[0] in X.columns)` handles it.
            # However, with the fixed preprocessor, it should just be `col in X.columns`.
            
            # Use X[NUMERICAL_FEATURES].values directly as the preprocessor already ensures they exist and are named correctly
            self.all_numerical_features = torch.tensor(X[NUMERICAL_FEATURES].values, dtype=torch.float32)
            global_logger.debug(f"      All Numerical: {self.all_numerical_features.shape=} {self.all_numerical_features.dtype=} {self.all_numerical_features.ndim=}")
        else:
            self.all_numerical_features = torch.empty((len(X), 0), dtype=torch.float32) # Empty tensor if no numerical
            global_logger.debug(f"      No numerical features found. Empty tensor created: {self.all_numerical_features.shape=}")

        # Process Labels
        if y is not None:
            global_logger.debug(" - Converting labels to PyTorch tensors")
            # Ensure labels are float32 for BCEWithLogitsLoss or Long for CrossEntropyLoss
            # Assuming binary classification for now (float32 for BCEWithLogitsLoss)
            self.labels = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1) # Add a dimension for target if needed (e.g., for BCEWithLogitsLoss)
            global_logger.debug(f"      Labels: {self.labels.shape=} {self.labels.dtype=} {self.labels.ndim=}")
        else:
            self.labels = None


        # Basic consistency check for number of samples across all data types
        num_samples = len(X)
        if (self.all_categorical_features.size(0) != num_samples and self.all_categorical_features.size(1) > 0) or \
           (self.all_numerical_features.size(0) != num_samples and self.all_numerical_features.size(1) > 0):
            raise ValueError(f"Mismatch in number of samples for combined features: {num_samples}")
        for key, tensor in self.text_features.items():
            if tensor.size(0) != num_samples:
                raise ValueError(f"Mismatch in number of samples for text feature '{key}': {tensor.size(0)} vs {num_samples}")
        if self.labels is not None and self.labels.size(0) != num_samples:
             raise ValueError(f"Mismatch in number of samples for labels: {self.labels.size(0)} vs {num_samples}")
        
        global_logger.debug("Dataset initialization complete.")

    def __len__(self):
        # Return the number of samples
        if self.labels is not None:
            return len(self.labels)
        elif self.text_features:
            return len(next(iter(self.text_features.values())))
        elif self.all_categorical_features.size(1) > 0:
            return len(self.all_categorical_features)
        elif self.all_numerical_features.size(1) > 0:
            return len(self.all_numerical_features)
        return 0 # Should not happen if X is not empty

    def __getitem__(self, idx):
        # 1. Text Features (as Dict[str, Tensor])
        text_inputs_dict = {name: self.text_features[name][idx]
                            for name in self.text_features.keys()}

        # 2. Categorical Features (as single Tensor)
        categorical_inputs_tensor = self.all_categorical_features[idx]

        # 3. Numerical Features (as single Tensor)
        numerical_inputs_tensor = self.all_numerical_features[idx]

        if self.labels is not None:
            label = self.labels[idx]
            return text_inputs_dict, categorical_inputs_tensor, numerical_inputs_tensor, label
        else:
            return text_inputs_dict, categorical_inputs_tensor, numerical_inputs_tensor