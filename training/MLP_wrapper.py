import torch
import torch.nn as nn
import torch.nn.functional as F 

from typing import List, Dict, Tuple

from loggers import global_logger, evaluation_logger

from base_deep_learner import BaseDeepLearningClassifier

# Fully Connected Neural Network Architecture 
class MultiInputMLPClassifier(nn.Module):
    """
    Defines a multi-input MLP architecture.
    """
    
    def __init__(self,  
                 text_specs: Dict[str, Tuple[int, int]],         # e.g., {'query': (10000, 50)}
                                                                 # where Key -> (Vocab Size, Embedding Dim)
                 categorical_specs: Dict[str, Tuple[int, int]],  # e.g., {'mehod': (150, 10)}
                                                                 # where Key -> (Cardinality, Embedding Dim)
                 num_numerical_features: int,
                 numerical_hidden_size: int,
                 hidden_size: int, 
                 num_classes: int,
                 dropout_rate: float = 0.5
                  ):
        super().__init__()

        
        # --- Text Embeddings (one branch per text feature with its own dimension) ---
        self.text_branches = nn.ModuleDict({
            name: nn.Embedding(vocab_size, embed_dim)
            for name, (vocab_size, embed_dim) in text_specs.items()
        })

        # --- Categorical Embeddings (one branch per categorical feature with its own dimension) ---
        self.cat_branches = nn.ModuleDict({
            name: nn.Embedding(cardinality, embed_dim)
            for name, (cardinality, embed_dim) in categorical_specs.items()
        })

        # --- Numerical Features ---
        self.num_branch = nn.Linear(num_numerical_features, numerical_hidden_size)

        # --- Fusion + Final Classifier ---
        text_fusion_dim = sum(embed_dim for _, embed_dim in text_specs.values())
        cat_fusion_dim = sum(embed_dim for _, embed_dim in categorical_specs.values())
        
        fusion_dim = text_fusion_dim + cat_fusion_dim + numerical_hidden_size
                
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, text_inputs, categorical_inputs, numerical_inputs):
        # --- Text Embeddings ---
        global_logger.debug(f"--- Text Embeddings ---")
        # Text: embed and mean-pool each feature
        text_outputs = []
        for i, (name, embedding_layer) in enumerate(self.text_branches.items()):
            embed = embedding_layer(text_inputs[i])  # (batch_size, seq_len, embed_dim)
            pooled = embed.mean(dim=1)  # (batch_size, embed_dim)
            global_logger.debug(f"{name=}, in: {text_inputs[i].shape=} out: {embed.shape=} outout: {pooled.shape=}")
            text_outputs.append(pooled)
        
        # --- Categorical Embeddings ---
        global_logger.debug(f"--- Categorical Embeddings ---")
        cat_outputs = []
        for i, (name, embedding_layer) in enumerate(self.cat_branches.items()):
            # categorical_inputs[i]: (batch_size, 1) or (batch_size,)
            embed = embedding_layer(categorical_inputs[i])  # (batch_size, embed_dim)
            global_logger.debug(f"{name=}, in: {categorical_inputs[i].shape=} out: {embed.shape=}")
            cat_outputs.append(embed)
        
        # --- Numerical Inputs ---
        global_logger.debug(f"--- Numerical Embeddings ---")
        # Each numerical_inputs[i]: (batch_size,)
        # Stack them to shape: (batch_size, num_numerical_features)
        num_tensor = torch.stack(numerical_inputs, dim=1).float()  # shape: (batch_size, num_numerical_features)
        num_output = F.relu(self.num_branch(num_tensor))  # (batch_size, numerical_hidden_size)
        global_logger.debug(f"Numerical, in: {num_tensor.shape=} out: {num_output.shape=}")

        # Combine all features
        global_logger.debug(f"--- Combined Embeddings ---")
        all_features = torch.cat(text_outputs + cat_outputs + [num_output], dim=1)
        global_logger.debug(f"Combined:{all_features.shape=} ")

        # Final layers
        global_logger.debug(f"--- Final Layers ---")
        logits = self.fusion(all_features)
        global_logger.debug(f"Logits:{logits.shape=} ")
        return logits

# Wrapper Class for PyTorch MLP to mimic Scikit-learn API
class PyTorchMLPClassifier(BaseDeepLearningClassifier):
    """
    A wrapper class for a PyTorch Fully Connected Neural Network (MLP)
    to enable its use with scikit-learn's GridSearchCV or RandomizedSearchCV.
    """
    
    def __init__(self, num_classes,
                 text_embed_dims, # e.g., {'RequestURIPath': 50}
                                                  # where Key ->  Embedding Dim
                 categorical_embed_dims,  # e.g., {'RequestMethod': 3}
                                                                 # where Key -> Embedding Dim):
                 
                 numerical_hidden_size,
                 preprocessor,
                 # Explicitly listed parameters inherited from BasePyTorchClassifier for scikit-learn's get_params()
                 learning_rate=0.001, epochs=50, batch_size=32, random_state=None, 
                 optimizer_type='adam',  optimizer_params=None,
                 loss_type='CrossEntropyLoss',  loss_params=None,                   
                 hidden_size=64, dropout_rate=0.6,
                  
                 ):
        """
        Initializes the PyTorch MLP classifier.
        All parameters are explicitly listed for scikit-learn compatibility.
        """
        # Pass some parameters to the superclass constructor
        super().__init__(
            learning_rate=learning_rate, 
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            optimizer_type=optimizer_type,
            optimizer_params=optimizer_params,
            loss_type=loss_type,
            loss_params=loss_params,
            num_classes=num_classes,
            preprocessor=preprocessor,
            text_embed_dims=text_embed_dims,
           categorical_embed_dims=categorical_embed_dims,
           numerical_hidden_size=numerical_hidden_size
        )
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        

        # It's good practice to also store the inherited parameters as attributes in the child class
        # to ensure they are properly recognized by BaseEstimator.get_params() for tuning.
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params
        self.loss_type = loss_type
        self.loss_params = loss_params

        self.num_classes = num_classes
        self.text_embed_dims = text_embed_dims
        self.categorical_embed_dims = categorical_embed_dims
        self.numerical_hidden_size = numerical_hidden_size
        self.preprocessor = preprocessor
        


    def _build_model_architecture(self):
        """
        Builds the specific MLP architecture for this classifier.
        """
    


        return MultiInputMLPClassifier( 
                 self.text_specs,
                 self.categorical_specs,
                 self.num_numerical_features,
                 self.numerical_hidden_size,
                 self.hidden_size, 
                 self.num_classes,
                 self.dropout_rate
                   )

