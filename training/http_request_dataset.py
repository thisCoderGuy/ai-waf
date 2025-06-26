import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import numpy as np

from config import (
    TEXT_FEATURES, CATEGORICAL_FEATURES,  NUMERICAL_FEATURES
)
from loggers import global_logger

class HTTPRequestMultiInputDataset(Dataset):
    def __init__(self, X, y):
        """
        Initializes a multi input dataset from preprocessed features.

        Args:
            X (np.array or sparse matrix): Preprocessed training features.
            y (np.array): Training labels.
        Returns:
            self: The dataset
        """

        self.labels = None
        self.text_features = OrderedDict()
        self.cat_features = OrderedDict()
        self.num_features = OrderedDict()

        global_logger.debug("Building Multi Input Dataset")
        # Convert np arrays to PyTorch tensors
        global_logger.debug(" - Converting text columns (np arrays) to PyTorch tensors")
        for text_feature in TEXT_FEATURES:
            # Collect all columns that start with the text_feature prefix
            prefix = f"{text_feature}_"
            text_columns = [col for col in X.columns if col.startswith(prefix)]
            feature_tensor = torch.tensor(X[text_columns].values, dtype=torch.long)            
            self.text_features[text_feature] = feature_tensor
            global_logger.debug(f"      {text_feature=}: {feature_tensor.shape=} {feature_tensor.dtype=} {feature_tensor.ndim=}")
            
        global_logger.debug(" - Converting categorical columns to PyTorch tensors")
        for categorical_feature in CATEGORICAL_FEATURES:
            categorical_tensor = torch.tensor(X[categorical_feature].values, dtype=torch.long)
            self.cat_features[categorical_feature] = categorical_tensor           
            global_logger.debug(f"      {categorical_feature=}: {categorical_tensor.shape=} {categorical_tensor.dtype=} {categorical_tensor.ndim=}")
            
            
        global_logger.debug("- Converting numerical columns to PyTorch tensors")
        for numerical_feature in NUMERICAL_FEATURES:
            numerical_tensor = torch.tensor(X[numerical_feature].values, dtype=torch.float32)
            self.num_features[numerical_feature] = numerical_tensor
            global_logger.debug(f"      {numerical_feature=}: {numerical_tensor.shape=} {numerical_tensor.dtype=} {numerical_tensor.ndim=}")
            

        num_text_features = len(self.text_features)
        num_cat_features = len(self.cat_features)
        num_num_features = len(self.num_features)
        global_logger.debug(f"{num_text_features=}")
        global_logger.debug(f"{num_cat_features=}")
        global_logger.debug(f"{num_num_features=}")

        if y is not None:
            global_logger.debug(" - Converting labels to PyTorch tensors")
            self.labels = torch.tensor(y.values, dtype=torch.float32)
            global_logger.debug(f"      Labels: {self.labels.shape=} {self.labels.dtype=} {self.labels.ndim=}")

        num_samples = self.__len__()
        for collection in [self.text_features, self.cat_features, self.num_features]:
            for key, tensor in collection.items():
                if tensor.size(0) != num_samples:
                    raise ValueError(f"Mismatch in number of samples for feature '{key}': {tensor.size(0)} vs {num_samples}")

    def __len__(self):
         return (len(self.labels) if self.labels is not None else
                len(next(iter(self.text_features.values()))) if self.text_features else
                len(next(iter(self.cat_features.values()))) if self.cat_features else
                len(next(iter(self.num_features.values()))) if self.num_features else 0)

    def __getitem__(self, idx):
        # Retrieve text features in the order of TEXT_FEATURES
        # which should match the order of text_specs.keys()
        text_inputs_list = [self.text_features[name][idx]
                            for name in self.text_features.keys()]

        # Retrieve categorical features in the order of CATEGORICAL_FEATURES
        # which should match the order of categorical_specs.keys()
        categorical_inputs_list = [self.cat_features[name][idx]
                                   for name in self.cat_features.keys()]

        # Retrieve numerical features in the order they were processed NUMERICAL_FEATURES
        numerical_inputs_list = [self.num_features[name][idx]
                                 for name in self.num_features.keys()]

        if self.labels is not None:
            label = self.labels[idx]
            return text_inputs_list, categorical_inputs_list, numerical_inputs_list, label
        else:
            return text_inputs_list, categorical_inputs_list, numerical_inputs_list






