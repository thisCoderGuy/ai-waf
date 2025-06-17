import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder 
from scipy.sparse import issparse
from base_deep_learner import BaseDeepLearningClassifier


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_size):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters, kernel_size=filter_size)
        self.fc1 = nn.Linear(n_filters, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x.long())
        x = x.permute(0, 2, 1) # Conv1d expects (batch, channels, seq_len)
        x = F.relu(self.conv(x))
        x = F.max_pool1d(x, kernel_size=x.shape[2]).squeeze(2) # Global max pooling
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Wrapper Class for PyTorch MLP to mimic Scikit-learn API
class CNNClassifier(BaseDeepLearningClassifier):
    
    def __init__(self, hidden_size=64, 
                 # Explicitly listed parameters inherited from BasePyTorchClassifier for scikit-learn's get_params()
                 learning_rate=0.001, epochs=50, batch_size=32, random_state=None, verbose=False,
                 optimizer_type='adam',  optimizer_params=None,
                 loss_type='CrossEntropyLoss',  loss_params=None):
        """
        Initializes the PyTorch MLP classifier.
        All parameters are explicitly listed for scikit-learn compatibility.
        """
        # Pass all parameters to the superclass constructor
        super().__init__(
            learning_rate=learning_rate, 
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            verbose=verbose,
            optimizer_type=optimizer_type,
            optimizer_params=optimizer_params,
            loss_type=loss_type,
            loss_params=loss_params
        )
        
        self.hidden_size = hidden_size


        # It's good practice to also store the inherited parameters as attributes in the child class
        # to ensure they are properly recognized by BaseEstimator.get_params() for tuning.
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params
        self.loss_type = loss_type
        self.loss_params = loss_params


    def _build_model_architecture(self, input_size, num_classes):
        """
        Builds the specific MLP architecture for this classifier.
        """
        return CNN(input_size, self.hidden_size, num_classes)

