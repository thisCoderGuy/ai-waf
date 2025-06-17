import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder 
from scipy.sparse import issparse
from base_deep_learner import BaseDeepLearningClassifier

# Fully Connected Neural Network Architecture 
class MLP(nn.Module):
    """
    Defines a simple Fully Connected Neural Network (MLP) architecture.
    It consists of one hidden layer with ReLU activation and an output layer.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Wrapper Class for PyTorch MLP to mimic Scikit-learn API
class PyTorchMLPClassifier(BaseDeepLearningClassifier):
    """
    A wrapper class for a PyTorch Fully Connected Neural Network (MLP)
    to enable its use with scikit-learn's GridSearchCV or RandomizedSearchCV.
    """
    
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
        return MLP(input_size, self.hidden_size, num_classes)

