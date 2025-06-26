import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict # Used in CustomMultiInputDataset (if included)


from loggers import global_logger, evaluation_logger

from base_deep_learner import BaseDeepLearningClassifier

class MultiInputRNNClassifier(nn.Module):
    """
    Defines a multi-input RNN architecture.
    """
    def __init__(self,
                 text_specs: Dict[str, Tuple[int, int]],         # e.g., {'RequestURIQuery': (10000, 50)}
                                                                # where Key -> (Vocab Size, Embedding Dim)
                 categorical_specs: Dict[str, Tuple[int, int]], # e.g., {'RequestMethod': (150, 10)}
                                                                # where Key -> (Cardinality, Embedding Dim)
                 num_numerical_features: int,
                 numerical_hidden_size: int,
                 hidden_size: int,
                 num_classes: int,
                 text_rnn_configs: Dict[str, Dict[str, int]],   # e.g., {'RequestURIQuery': {'hidden_size': 128, 'num_layers': 1, 'bidirectional': False}}
                 rnn_type: str = 'GRU',                         # 'GRU' or 'LSTM'
                 dropout_rate: float = 0.5
                 ):
        super().__init__()

        # --- Text Embeddings (one branch per text feature with its own dimension) ---
        self.text_embedding_branches = nn.ModuleDict({
            name: nn.Embedding(vocab_size, embed_dim)
            for name, (vocab_size, embed_dim) in text_specs.items()
        })

        # --- Categorical Embeddings (one branch per categorical feature with its own dimension) ---
        self.cat_embedding_branches = nn.ModuleDict({
            name: nn.Embedding(cardinality, embed_dim)
            for name, (cardinality, embed_dim) in categorical_specs.items()
        })

        # --- Numerical Features ---
        self.num_linear_branch = nn.Linear(num_numerical_features, numerical_hidden_size)




        # --- RNN layers for text processing ---
        # Create a dictionary of RNN layers, one for each text feature.
        self.text_rnn_layers = nn.ModuleDict()
        self.text_rnn_configs = text_rnn_configs
        for name, (_, embed_dim) in text_specs.items():
            rnn_config = text_rnn_configs[name]
            rnn_hidden_size = rnn_config['hidden_size']
            num_layers = rnn_config.get('num_layers', 1)
            bidirectional = rnn_config.get('bidirectional', False)

            if rnn_type.upper() == 'GRU':
                self.text_rnn_layers[name] = nn.GRU(
                    input_size=embed_dim,
                    hidden_size=rnn_hidden_size,
                    num_layers=num_layers,
                    batch_first=True, # Input tensor shape: (batch_size, sequence_length, input_size)
                    bidirectional=bidirectional
                )
            elif rnn_type.upper() == 'LSTM':
                self.text_rnn_layers[name] = nn.LSTM(
                    input_size=embed_dim,
                    hidden_size=rnn_hidden_size,
                    num_layers=num_layers,
                    batch_first=True, # Input tensor shape: (batch_size, sequence_length, input_size)
                    bidirectional=bidirectional
                )
            else:
                raise ValueError(f"Unsupported RNN type: {rnn_type}. Choose 'GRU' or 'LSTM'.")
        self.rnn_type = rnn_type # Store for internal reference if needed later

        # --- Calculate fusion dimension ---
        # Get the total dimension from all categorical feature embeddings.
        cat_fusion_dim = sum(embed_dim for _, embed_dim in categorical_specs.values())

        # Get the total dimension from all text features.
        # The output dimension from each RNN branch will be hidden_size * (2 if bidirectional else 1).
        text_fusion_dim = 0
        for name in text_specs.keys():
            rnn_config = text_rnn_configs[name]
            rnn_hidden_size = rnn_config['hidden_size']
            bidirectional = rnn_config.get('bidirectional', False)
            text_fusion_dim += rnn_hidden_size * (2 if bidirectional else 1)

        # The final fusion dimension is the sum of the text, categorical, and numerical feature dimensions.
        fusion_dim = text_fusion_dim + cat_fusion_dim + numerical_hidden_size

        # --- Fusion block (standard MLP) ---
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.ReLU(),           # activation
            nn.Dropout(dropout_rate), # regularization
            nn.Linear(hidden_size, num_classes)
        )





    def forward(self, text_inputs: List[torch.Tensor],
                categorical_inputs: List[torch.Tensor],
                numerical_inputs: List[torch.Tensor]):
        """
        Forward pass for the MultiInputRNNClassifier.

        Args:
            text_inputs (List[torch.Tensor]): A list of tensors, where each tensor corresponds to a text feature.
                                              The order of tensors in the list should match the order of keys
                                              in `text_specs` during initialization.
                                              Each tensor has shape (batch_size, sequence_length).
            categorical_inputs (List[torch.Tensor]): A list of tensors, where each tensor corresponds to a
                                                     categorical feature. The order should match `categorical_specs`.
                                                     Each tensor has shape (batch_size,).
            numerical_inputs (List[torch.Tensor]): A list of tensors, where each tensor corresponds to a
                                                   single numerical feature.
                                                   Each tensor has shape (batch_size,).

        Returns:
            torch.Tensor: The output logits (batch_size, num_classes).
        """
        global_logger.debug(f"--- Text Embeddings and RNN Processing ---")
        processed_text_features = []
        for i, name in enumerate(self.text_embedding_branches.keys()):
            input_tensor = text_inputs[i] # Get the correct tensor from the list based on index
            embedded_text = self.text_embedding_branches[name](input_tensor) # (batch_size, sequence_length, embed_dim)

            # Pass through RNN layer
            # output: (batch_size, seq_len, num_directions * hidden_size)
            # h_n: (num_layers * num_directions, batch_size, hidden_size) - for GRU
            # (h_n, c_n): similar for LSTM
            rnn_output, hidden_state = self.text_rnn_layers[name](embedded_text)

            # Extract the last hidden state of the last layer
            # For bidirectional, we concatenate the last forward and backward hidden states
            rnn_config = self.text_rnn_configs[name] # Access directly using self.text_rnn_configs
            num_directions = 2 if rnn_config.get('bidirectional', False) else 1
            num_layers = rnn_config.get('num_layers', 1)
            rnn_hidden_size = rnn_config['hidden_size']

            if num_directions == 1: # Unidirectional
                # hidden_state is (num_layers, batch_size, hidden_size) for GRU/LSTM h_n
                # We need the last layer's hidden state: hidden_state[-1, :, :]
                pooled_output = hidden_state[-1, :, :] # (batch_size, hidden_size)
            else: # Bidirectional
                # For bidirectional, h_n[-2, :, :] is the last forward hidden state
                # and h_n[-1, :, :] is the last backward hidden state.
                # We concatenate them.
                pooled_output = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1) # (batch_size, 2 * hidden_size)

            processed_text_features.append(pooled_output)
            global_logger.debug(f"Text feature '{name}', input shape: {input_tensor.shape}, "
                                f"embedded shape: {embedded_text.shape}, "
                                f"pooled output shape: {pooled_output.shape}")

        # --- Categorical Embeddings ---
        global_logger.debug(f"--- Categorical Embeddings ---")
        cat_outputs = []
        for i, name in enumerate(self.cat_embedding_branches.keys()):
            input_tensor = categorical_inputs[i]
            embed = self.cat_embedding_branches[name](input_tensor) # (batch_size, embed_dim)
            global_logger.debug(f"{name=}, in: {input_tensor.shape=} out: {embed.shape=}")
            cat_outputs.append(embed)

        # --- Numerical Inputs ---
        global_logger.debug(f"--- Numerical Linear layer ---")
        num_tensor = torch.stack(numerical_inputs, dim=1).float() # shape: (batch_size, num_numerical_features)
        num_output = F.relu(self.num_linear_branch(num_tensor)) # (batch_size, numerical_hidden_size)
        global_logger.debug(f"Numerical, in: {num_tensor.shape=} out: {num_output.shape=}")

        # Combine all features
        global_logger.debug(f"--- Combined Embeddings ---")
        all_features = torch.cat(processed_text_features + cat_outputs + [num_output], dim=1)
        global_logger.debug(f"Combined features shape: {all_features.shape}")

        # Final layers (fusion block)
        global_logger.debug(f"--- Final Layers ---")
        logits = self.fusion(all_features)
        global_logger.debug(f"Logits shape: {logits.shape}")
        return logits


class RNNClassifier(BaseDeepLearningClassifier):
    """
    A wrapper class for a PyTorch RNN (Recurrent Neural Network)
    to enable its use with scikit-learn's GridSearchCV or RandomizedSearchCV.
    """

    def __init__(self, num_classes,
                 text_embed_dims: Dict[str, int],      # e.g., {'RequestURIPath': 50} where Key -> Embedding Dim
                 categorical_embed_dims: Dict[str, int], # e.g., {'RequestMethod': 3} where Key -> Embedding Dim
                 numerical_hidden_size: int,
                 preprocessor,
                 text_rnn_configs: Dict[str, Dict[str, int]], # e.g., {'RequestURIQuery': {'hidden_size': 128, 'num_layers': 1, 'bidirectional': False}}
                 rnn_type: str = 'GRU',
                 # Explicitly listed parameters inherited from BasePyTorchClassifier for scikit-learn's get_params()
                 learning_rate: float = 0.001, epochs: int = 50, batch_size: int = 32, random_state: Optional[int] = None,
                 optimizer_type: str = 'adam', optimizer_params: Optional[Dict] = None,
                 loss_type: str = 'CrossEntropyLoss', loss_params: Optional[Dict] = None,
                 hidden_size: int = 64, dropout_rate: float = 0.6
                 ):
        """
        Initializes the PyTorch RNN classifier.
        All parameters are explicitly listed for scikit-learn compatibility.
        """
        # Pass relevant parameters to the superclass constructor
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
        self.text_rnn_configs = text_rnn_configs
        self.rnn_type = rnn_type

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
        Builds the specific RNN architecture for this classifier.
        """
        return MultiInputRNNClassifier(
            text_specs=self.text_specs,
            categorical_specs=self.categorical_specs,
            num_numerical_features=self.num_numerical_features,
            numerical_hidden_size=self.numerical_hidden_size,
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            text_rnn_configs=self.text_rnn_configs,
            rnn_type=self.rnn_type,
            dropout_rate=self.dropout_rate
        )

