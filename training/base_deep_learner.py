import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder 
from scipy.sparse import issparse
from torch.utils.data import TensorDataset, DataLoader


class BaseDeepLearningClassifier(BaseEstimator, ClassifierMixin):
    """
    A base class for wrapping PyTorch deep neural networks to mimic scikit-learn's API.
    Handles common functionalities like device setup, random seeds, training loop,
    prediction, and dynamic creation of loss function and optimizer.
    """
    def __init__(self,  num_classes,
            text_embed_dims,
            categorical_embed_dims,
            numerical_hidden_size,
            preprocessor,
                 learning_rate=0.001, epochs=50, batch_size=32,
                 random_state=None, verbose=False,
                 optimizer_type='adam',  optimizer_params=None,
                 loss_type='CrossEntropyLoss',  loss_params=None):
        """
        Initializes the base PyTorch classifier.

        Args:
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
            batch_size (int): Size of training mini-batches.
            random_state (int): Seed for reproducibility.
            verbose (bool): Whether to print training progress.
            optimizer_params (dict): Dictionary with 'type' and 'hyperparameters' for the optimizer.
                                     If None, defaults from config.py will be used.
            loss_params (dict): Dictionary with 'type' and 'hyperparameters' for the loss function.
                                If None, defaults from config.py will be used.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.label_encoder = LabelEncoder()
        self.criterion = None
        self.loss_type = loss_type
        self.loss_params = loss_params
        self.learning_rate = learning_rate
        self.optimizer = None
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params

        self.num_classes = num_classes
        self.numerical_hidden_size = numerical_hidden_size
        
        
        categorical_embed_dims = categorical_embed_dims
        
        self.text_specs = {}
        for text_feature in text_embed_dims:
            vocab_size = preprocessor[f"{text_feature}_vocab_size"]
            self.text_specs[text_feature] = (vocab_size, text_embed_dims[text_feature])
        

        self.categorical_specs = {}
        for categorical_feature in categorical_embed_dims:
            cardinality = preprocessor[f"{categorical_feature}_cardinality"] 
            self.categorical_specs[categorical_feature] = (cardinality, categorical_embed_dims[categorical_feature])
                

        self.num_numerical_features =  preprocessor["num_numerical_features"]



        # Determine the device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"Using device: {self.device}")

        # Set random seeds for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if self.device.type == 'cuda':
                torch.cuda.manual_seed(self.random_state)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False # Set to False for reproducibility

    def _build_model_architecture(self):
        """
        Abstract method to be implemented by subclasses.
        This method should return the specific nn.Module for the classifier.
        """
        raise NotImplementedError("Subclasses must implement _build_model_architecture method.")

    def _build_model_components(self):
        """
        Builds the PyTorch model architecture, loss function, and optimizer.
        """
        self.model = self._build_model_architecture()
        self.model.to(self.device) # <--- Move the model to the selected device

        # Dynamically create loss function
        if self.loss_type == 'CrossEntropyLoss':
            # Handle 'weight' parameter if present: it needs to be a torch.Tensor
            if 'weight' in self.loss_params and isinstance(self.loss_params['weight'], list):
                self.loss_params['weight'] = torch.tensor(self.loss_params['weight'], dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(**self.loss_params)
        elif self.loss_type == 'BCEWithLogitsLoss':
            # BCEWithLogitsLoss is typically used for binary classification, might need 'pos_weight'
            if 'pos_weight' in self.loss_params and isinstance(self.loss_params['pos_weight'], list):
                self.loss_params['pos_weight'] = torch.tensor(self.loss_params['pos_weight'], dtype=torch.float32).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(**self.loss_params)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

       
        # Dynamically create optimizer        
        # Use learning_rate from init if provided directly (for backward compatibility),
        # otherwise use lr from optimizer_hyperparams.
        # Priority: init's learning_rate > optimizer_params['hyperparameters']['lr']
        if self.learning_rate is not None:
            self.optimizer_params['lr'] = self.learning_rate
        elif 'lr' not in self.optimizer_params:
             # Fallback if no lr specified anywhere
             self.optimizer_params['lr'] = 0.001

        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), **self.optimizer_params)
        elif self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), **self.optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")



    def fit(self, X, y):
        """
        Trains the PyTorch model.

        Args:
            X (np.array or sparse matrix): Training features.
            y (np.array): Training labels.
        Returns:
            self: The trained classifier.
        """
        


        ##########################
        # text_vocab_sizes: Dict[str, int], # e.g., {'request_uri_path': 50}  where Key -> Vocab Size
        # categorical_cardinalities: Dict[str,  int]  # e.g., {'request_method': 10} where Key -> Cardinality 
        # num_numerical_features: int

        text_vocab_sizes  = 
        categorical_cardinalities =
        num_numerical_features =

        # Build the model components if not already built
        if self.model is None:
            self._build_model_components()



        # Convert to PyTorch tensors
        X_text_tensors = [
            torch.tensor(X['request_uri_path'], dtype=torch.long),
            torch.tensor(X['request_uri_query'], dtype=torch.long),
            torch.tensor(X['request_body'], dtype=torch.long),
            torch.tensor(X['user_agent'], dtype=torch.long)
        ]
        X_categorical_tensor = torch.tensor(X['categorical'], dtype=torch.long)
        X_numerical_tensor = torch.tensor(X['numerical'], dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)

        if self.verbose:
            print(f"Starting training on device: {self.device}")
            print(f"  Model type: {self.__class__.__name__}")
            print(f"  Optimizer: {self.optimizer_params['type']} (lr={self.optimizer.param_groups[0]['lr']})")
            print(f"  Loss: {self.loss_params['type']}")
            print(f"  Epochs: {self.epochs}, Batch Size: {self.batch_size}")
            print(f"  Device: {self.device}")


        # Create DataLoader for batching
        # Combine everything into one dataset
        train_dataset = TensorDataset(
            X_text_tensors[0],
            X_text_tensors[1],
            X_text_tensors[2],
            X_text_tensors[3],
            X_categorical_tensor,
            X_numerical_tensor,
            y_tensor
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # TODO
        # L1 regularization
        # l1_lambda = 0.001
        # L2 regularization , same as weight_decay param in Adam Optimizer
        # l2_lambda = 0.001

        for epoch in range(self.epochs):
            self.model.train() # Set model to training mode
            total_loss = 0

            for batch in train_loader:
                # Unpack and move inputs to device
                path_tensor, query_tensor, body_tensor, ua_tensor, cat_tensor, num_tensor, labels = [
                    x.to(self.device) for x in batch
                ]

                # Forward pass
                logits = self.model(
                    text_inputs=[path_tensor, query_tensor, body_tensor, ua_tensor],
                    cat_inputs=cat_tensor,
                    num_input=num_tensor
                )  # shape: (batch_size, 2)   # 2 output classes

                loss = self.criterion(logits, labels.long())  # Use CrossEntropyLoss
                 # TODO
                # Elastic Net: L1 + L2 regularization
                # l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                # l2_norm = sum(p.pow(2).sum() for p in self.model.parameters())
                # loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")
     

        
        return self

    def predict(self, X):
        """
        Predicts class labels for the input data.

        Args:
            X (np.array or sparse matrix): Features to predict on.
        Returns:
            np.array: Predicted class labels.
        """
        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        self.model.eval() # Set model to evaluation mode
        X_tensor = torch.tensor(X_dense, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted_encoded = torch.max(outputs.data, 1)

        # Move predictions back to CPU for numpy conversion and decoding
        return self.label_encoder.inverse_transform(predicted_encoded.cpu().numpy())

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input data.

        Args:
            X (np.array or sparse matrix): Features to predict on.
        Returns:
            np.array: Predicted class probabilities.
        """
        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        self.model.eval() # Set model to evaluation mode
        X_tensor = torch.tensor(X_dense, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        # Move probabilities back to CPU for numpy conversion
        return probabilities.cpu().numpy()
    