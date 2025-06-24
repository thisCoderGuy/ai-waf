import torch
import torch.nn as nn
import torch.nn.functional as F 

from typing import Dict, Tuple

from loggers import global_logger, evaluation_logger


from base_deep_learner import BaseDeepLearningClassifier
class MultiInputCNNClassifier(nn.Module):
    def __init__(self,
                 text_specs: Dict[str, Tuple[int, int]],         # e.g., {'RequestURIQuery': (10000, 50)}
                                                                 # where Key -> (Vocab Size, Embedding Dim)
                 categorical_specs: Dict[str, Tuple[int, int]],  # e.g., {'RequestMethod': (150, 10)}
                                                                 # where Key -> (Cardinality, Embedding Dim)
                 num_numerical_features: int,
                 numerical_hidden_size: int,  
                 hidden_size: int,                
                 num_classes: int,
                 text_cnn_configs: Dict[str, Dict[str, int]],    # e.g., {'RequestURIQuery': {'n_filters': 96, 'filter_size': 5}}
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





        # --- CNN layers for text processing ---
        # Create a dictionary of convolution layers, one for each text feature.
        self.text_conv_layers = nn.ModuleDict({
            name: nn.Conv1d(
                in_channels=embed_dim,                                  # Input channels = embedding dim
                out_channels=text_cnn_configs[name]['n_filters'],       # Use the specific n_filters for this feature
                kernel_size=text_cnn_configs[name]['filter_size']       # Use the specific filter_size for this feature
            )
            for name, (_, embed_dim) in text_specs.items()
        })

        

        # --- Calculate fusion dimension ---
        # Get the total dimension from all categorical feature embeddings.
        cat_fusion_dim = sum(embed_dim for _, embed_dim in categorical_specs.values())

        # Get the total dimension from all text features. After each text feature's
        # embeddings are processed by a Conv1d layer and pooled, they will each have a size of n_filters.
        text_fusion_dim = sum(config['n_filters'] for config in text_cnn_configs.values())

        # The final fusion dimension is the sum of the text, categorical, and numerical feature dimensions.
        fusion_dim = text_fusion_dim + cat_fusion_dim + numerical_hidden_size

        # --- Fusion block (standard MLP) ---
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )




        

    def forward(self, text_inputs: Dict[str, torch.Tensor],
                categorical_inputs: Dict[str, torch.Tensor],
                numerical_inputs: torch.Tensor):
        """
        Forward pass for the MultiInputCNNClassifier.

        Args:
            text_inputs (Dict[str, torch.Tensor]): A dictionary where keys are text feature names
                                                  and values are their corresponding input tensors
                                                  (batch_size, sequence_length).
            categorical_inputs (Dict[str, torch.Tensor]): A dictionary where keys are categorical feature names
                                                          and values are their corresponding input tensors
                                                          (batch_size,).
            numerical_inputs (torch.Tensor): A tensor containing numerical features (batch_size, num_numerical_features).

        Returns:
            torch.Tensor: The output logits (batch_size, num_classes).
        """
        processed_text_features = []
        # Process each text input
        for name, input_tensor in text_inputs.items():
            # Get embeddings: (batch_size, sequence_length, embed_dim)
            embedded_text = self.text_embedding_branches[name](input_tensor)
            # Transpose for Conv1d: (batch_size, embed_dim, sequence_length)
            embedded_text = embedded_text.permute(0, 2, 1)
            # Apply convolution
            conv_output = self.text_conv_layers[name](embedded_text)
            # Apply ReLU activation
            conv_output = F.relu(conv_output)
            # Apply max pooling over the sequence dimension
            # Max pooling output will be (batch_size, n_filters, 1), squeeze to remove last dim
            pooled_output = F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)
            processed_text_features.append(pooled_output)






            
        # --- Text Embeddings ---
        global_logger.debug(f"--- Text Embeddings and Convolution---")
        # Text: embed convolve and mean-pool each feature
        processed_text_features = []
        # Process each text input
        # We iterate over the keys of text_embedding_branches to maintain the correct mapping
        # between feature names and the list of input tensors.
        for i, name in enumerate(self.text_embedding_branches.keys()):
            input_tensor = text_inputs[i] # Get the tensor from the list based on index
            # Get embeddings: (batch_size, sequence_length, embed_dim)
            embedded_text = self.text_embedding_branches[name](input_tensor)
            # Transpose for Conv1d: (batch_size, embed_dim, sequence_length)
            embedded_text = embedded_text.permute(0, 2, 1)
            # Apply convolution
            conv_output = self.text_conv_layers[name](embedded_text)
            # Apply ReLU activation
            conv_output = F.relu(conv_output)
            # Apply max pooling over the sequence dimension
            # Max pooling output will be (batch_size, n_filters, 1), squeeze to remove last dim
            pooled_output = F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)
            processed_text_features.append(pooled_output)



        # Process each text input
        for name, input_tensor in text_inputs:
            # Get embeddings: (batch_size, sequence_length, embed_dim)
            embedded_text = self.text_embedding_branches[name](input_tensor)
            # Transpose for Conv1d: (batch_size, embed_dim, sequence_length)
            embedded_text = embedded_text.permute(0, 2, 1)
            # Apply convolution
            conv_output = self.text_conv_layers[name](embedded_text)
            # Apply ReLU activation
            conv_output = F.relu(conv_output)
            # Apply max pooling over the sequence dimension
            # Max pooling output will be (batch_size, n_filters, 1), squeeze to remove last dim
            pooled_output = F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)
            processed_text_features.append(pooled_output)

        
        
        # --- Categorical Embeddings ---
        global_logger.debug(f"--- Categorical Embeddings ---")
        processed_categorical_features = []
        for name, input_tensor in categorical_inputs.items():
            # Get embeddings: (batch_size, embed_dim)
            embedded_cat = self.cat_embedding_branches[name](input_tensor)
            # categorical_inputs[i]: (batch_size, 1) or (batch_size,)
            #embed = embedding_layer(categorical_inputs[i])  # (batch_size, embed_dim)
            global_logger.debug(f"{name=}, in: {input_tensor.shape=} out: {embedded_cat.shape=}")
            processed_categorical_features.append(embedded_cat)
        
        # --- Numerical Inputs ---
        global_logger.debug(f"--- Numerical Linear layer ---")
        # Each numerical_inputs[i]: (batch_size,)
        # Stack them to shape: (batch_size, num_numerical_features)
        num_tensor = torch.stack(numerical_inputs, dim=1).float()  # shape: (batch_size, num_numerical_features)
        numerical_branch_output = F.relu(self.num_linear_branch(num_tensor))  # (batch_size, numerical_hidden_size)
        global_logger.debug(f"Numerical, in: {num_tensor.shape=} out: {numerical_branch_output.shape=}")

        # Combine all features by concatenating
        global_logger.debug(f"--- Combined Embeddings by concatenating ---")
        # Ensure that if any list is empty, it doesn't cause concatenation issues
        all_features = []
        if processed_text_features:
            all_features.append(torch.cat(processed_text_features, dim=1))
        if processed_categorical_features:
            all_features.append(torch.cat(processed_categorical_features, dim=1))
        all_features.append(numerical_branch_output)
        fused_features = torch.cat(all_features, dim=1)
        global_logger.debug(f"Combined:{all_features.shape=} ")

        # Final layers (fusion block)
        global_logger.debug(f"--- Final Layers ---")
        logits = self.fusion(fused_features)
        global_logger.debug(f"Logits:{logits.shape=} ")
        return logits

# Wrapper Class for PyTorch  to mimic Scikit-learn API
class CNNClassifier(BaseDeepLearningClassifier):
    
    def __init__(self, num_classes,
                 text_embed_dims, # e.g., {'RequestURIPath': 50}
                                                  # where Key ->  Embedding Dim
                 categorical_embed_dims,  # e.g., {'RequestMethod': 3}
                                                                 # where Key -> Embedding Dim):
                 
                 numerical_hidden_size,
                 preprocessor,
                 text_cnn_configs,
                 # Explicitly listed parameters inherited from BasePyTorchClassifier for scikit-learn's get_params()
                 learning_rate=0.001, epochs=50, batch_size=32, random_state=None, 
                 optimizer_type='adam',  optimizer_params=None,
                 loss_type='CrossEntropyLoss',  loss_params=None,                   
                 hidden_size=64, dropout_rate=0.6                
                 
                 ):
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
        self.text_cnn_configs = text_cnn_configs


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
        Builds the specific CNN architecture for this classifier.
        """
        return MultiInputCNNClassifier(
                text_specs=self.text_specs,
                categorical_specs=self.categorical_specs,
                num_numerical_features=self.num_numerical_features,
                numerical_hidden_size=self.numerical_hidden_size,
                hidden_size=self.hidden_size,
                num_classes=self.num_classes,
                text_cnn_configs=self.text_cnn_configs,
                dropout_rate=self.dropout_rate
            )

