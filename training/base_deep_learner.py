import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder 
from scipy.sparse import issparse

class BaseDeepLearningClassifier(BaseEstimator, ClassifierMixin):
    """
    A base class for wrapping PyTorch deep neural networks to mimic scikit-learn's API.
    Handles common functionalities like device setup, random seeds, training loop,
    prediction, and dynamic creation of loss function and optimizer.
    """
    def __init__(self, learning_rate=0.001, epochs=50, batch_size=32,
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

    def _build_model_architecture(self, input_size, num_classes):
        """
        Abstract method to be implemented by subclasses.
        This method should return the specific nn.Module for the classifier.
        """
        raise NotImplementedError("Subclasses must implement _build_model_architecture method.")

    def _build_model_components(self, input_size, num_classes):
        """
        Builds the PyTorch model architecture, loss function, and optimizer.
        """
        self.model = self._build_model_architecture(input_size, num_classes)
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
        # Convert sparse matrix to dense numpy array if necessary
        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        # Infer input_size and num_classes
        self.input_size = X_dense.shape[1] # Num columns, ie num features
        self.classes_ = self.label_encoder.fit(y).classes_
        self.num_classes = len(self.classes_)

        # Build the model components if not already built
        if self.model is None:
            self._build_model_components(self.input_size, self.num_classes)

        # Convert data to PyTorch tensors and move to device
        X_tensor = torch.tensor(X_dense, dtype=torch.float32).to(self.device)
        y_encoded = self.label_encoder.transform(y)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(self.device)

        self.model.train() # Set model to training mode

        if self.verbose:
            print(f"Starting training on device: {self.device}")
            print(f"  Model type: {self.__class__.__name__}")
            print(f"  Optimizer: {self.optimizer_params['type']} (lr={self.optimizer.param_groups[0]['lr']})")
            print(f"  Loss: {self.loss_params['type']}")
            print(f"  Epochs: {self.epochs}, Batch Size: {self.batch_size}")
            print(f"  Device: {self.device}")


        # Create DataLoader for batching
        train_data = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f'  Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
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