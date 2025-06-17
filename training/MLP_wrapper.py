import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder 
from scipy.sparse import issparse
from base_deep_learner import BaseDeepLearningClassifier

# Fully Connected Neural Network Architecture 
class MultiInputClassifier(nn.Module):
    """
    Defines a simple Fully Connected Neural Network (MLP) architecture.
    It consists of one hidden layer with ReLU activation and an output layer.
    """
    
    def __init__(self,  
                 vocab_sizes,           # dict of vocab sizes per text feature
                 text_embed_dim,       # embedding dims for text features
                 categorical_cardinalities,  # e.g., [5] for 1 categorical feature with 5 values
                 categorical_embedding_dim,
                 num_numerical_features,
                 numerical_hidden_size,
                 hidden_size, 
                 num_classes,
                 dropout_rate=0.5
                  ):
        super().__init__()

        # --- Text Embeddings (one branch per text feature) ---
        self.text_branches = nn.ModuleDict({
            name: nn.Embedding(vocab_size, text_embed_dim)
            for name, vocab_size in vocab_sizes.items()
        })

        # Categorical Embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_classes, categorical_embedding_dim)
            for num_classes in categorical_cardinalities
        ])

        # --- Numerical Features ---
        self.num_branch = nn.Linear(num_numerical_features, numerical_hidden_size)

        # --- Fusion + Final Classifier ---
        fusion_dim = len(vocab_sizes) * text_embed_dim + \
                     len(categorical_cardinalities) * categorical_embedding_dim + \
                     numerical_hidden_size  # from numerical branch
        
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, text_inputs: dict, categorical_inputs: torch.Tensor, numerical_inputs: torch.Tensor):
       # Text: embed and mean-pool each feature
        text_outputs = []
        for name, embed_layer in self.text_branches.items():
            x = embed_layer(text_inputs[name].long())  # (batch, seq_len, emb_dim)
            x = x.mean(dim=1)  # Mean pooling → (batch, emb_dim)
            text_outputs.append(x)

        # Categorical: embed and concat
        cat_outputs = [
            emb(categorical_inputs[:, i].long())  # (batch, emb_dim)
            for i, emb in enumerate(self.cat_embeddings)
        ]

        # Numerical: FC layer
        num_output = F.relu(self.num_branch(numerical_inputs.float()))  # (batch, 64)

        # Combine all features
        all_features = torch.cat(text_outputs + cat_outputs + [num_output], dim=1)

        # Final layers
        logits = self.fusion(all_features)
        return logits

# Wrapper Class for PyTorch MLP to mimic Scikit-learn API
class PyTorchMLPClassifier(BaseDeepLearningClassifier):
    """
    A wrapper class for a PyTorch Fully Connected Neural Network (MLP)
    to enable its use with scikit-learn's GridSearchCV or RandomizedSearchCV.
    """
    
    def __init__(self, hidden_size=64, dropout_rate=0.6,
                 # Explicitly listed parameters inherited from BasePyTorchClassifier for scikit-learn's get_params()
                 learning_rate=0.001, epochs=50, batch_size=32, random_state=None, verbose=False,
                 optimizer_type='adam',  optimizer_params=None,
                 loss_type='CrossEntropyLoss',  loss_params=None, num_classes=2,
                 text_embed_dim=32, categorical_embedding_dim=2, numerical_hidden_size=32):
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
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.text_embed_dim = text_embed_dim
        self.categorical_embedding_dim = categorical_embedding_dim
        self.numerical_hidden_size = numerical_hidden_size

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
        return MultiInputClassifier( vocab_sizes,           # dict of vocab sizes per text feature
                 self.text_embed_dim,       # embedding dims for text features
                 categorical_cardinalities,  # e.g., [5] for 1 categorical feature with 5 values
                 self.categorical_embedding_dim,
                 num_numerical_features,
                 self.numerical_hidden_size,
                 self.hidden_size, 
                 self.num_classes,
                 self.dropout_rate
                   )

