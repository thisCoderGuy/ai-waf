import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder 
from scipy.sparse import issparse

# Fully Connected Neural Network Architecture 
class SimpleFCNet(nn.Module):
    """
    Defines a simple Fully Connected Neural Network (MLP) architecture.
    It consists of one hidden layer with ReLU activation and an output layer.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleFCNet, self).__init__()
        # First fully connected (linear) layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function for the hidden layer
        self.relu = nn.ReLU()
        # Second fully connected (linear) layer, outputs raw scores (logits)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Wrapper Class for PyTorch MLP to mimic Scikit-learn API
class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    A wrapper class for a PyTorch Fully Connected Neural Network (MLP)
    to enable its use with scikit-learn's GridSearchCV or RandomizedSearchCV.
    """
    def __init__(self, input_size=None, hidden_size=64, num_classes=None,
                 learning_rate=0.001, epochs=50, batch_size=32,
                 random_state=None, verbose=False,
                 optimizer_params=None, loss_params=None):
        """
        Initializes the PyTorch MLP classifier.

        Args:
            input_size (int): Number of input features. Will be inferred if None during fit.
            hidden_size (int): Number of neurons in the hidden layer.
            num_classes (int): Number of output classes. Will be inferred if None during fit.
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
            batch_size (int): Size of training mini-batches.
            random_state (int): Seed for reproducibility.
            verbose (bool): Whether to print training progress.   
            optimizer_params (dict): Dictionary with 'type' and 'hyperparameters' for the optimizer.
            loss_params (dict): Dictionary with 'type' and 'hyperparameters' for the loss function.
       
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.label_encoder = LabelEncoder() # To handle potential string labels from data
         # Store optimizer and loss parameters
        # Use provided params or fall back to config.py defaults
        self.optimizer_params = optimizer_params if optimizer_params is not None else config.MLP_OPTIMIZER_PARAMS
        self.loss_params = loss_params if loss_params is not None else config.MLP_LOSS_PARAMS


        # Set random seeds for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            # Further seeds for CUDA if available
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def _build_model(self):
        """
        Builds the PyTorch SimpleFCNet model.
        Assumes input_size and num_classes are set (either during init or fit).
        """
        if self.input_size is None or self.num_classes is None:
            raise ValueError("input_size and num_classes must be set before building the model.")
        self.model = SimpleFCNet(self.input_size, self.hidden_size, self.num_classes)
      
      
          # Dynamically create loss function
        loss_type = self.loss_params.get('type', 'CrossEntropyLoss')
        loss_hyperparams = self.loss_params.get('hyperparameters', {})
        if loss_type == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss(**loss_hyperparams)
        elif loss_type == 'BCEWithLogitsLoss':
            self.criterion = nn.BCEWithLogitsLoss(**loss_hyperparams)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Dynamically create optimizer
        optimizer_type = self.optimizer_params.get('type', 'Adam')
        optimizer_hyperparams = self.optimizer_params.get('hyperparameters', {})

        # Use learning_rate from init if provided directly (for backward compatibility),
        # otherwise use lr from optimizer_hyperparams.
        # Priority: init's learning_rate > optimizer_params['hyperparameters']['lr']
        if self.learning_rate is not None:
            optimizer_hyperparams['lr'] = self.learning_rate
        elif 'lr' not in optimizer_hyperparams:
             # Fallback if no lr specified anywhere
             optimizer_hyperparams['lr'] = 0.001


        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_hyperparams)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), **optimizer_hyperparams)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def fit(self, X, y):
        """
        Trains the PyTorch MLP model.

        Args:
            X (np.array): Training features.
            y (np.array): Training labels.
        Returns:
            self: The trained classifier.
        """

        # Convert sparse matrix to dense numpy array if necessary
        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        # Infer input_size and num_classes if not provided during initialization
        self.input_size = X.shape[1]
        self.classes_ = self.label_encoder.fit(y).classes_
        self.num_classes = len(self.classes_)

        # Build the model if not already built
        if self.model is None:
            self._build_model()

        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X_dense, dtype=torch.float32)
        # Encode labels to numerical values (0, 1, ...) if they are not already
        y_encoded = self.label_encoder.transform(y)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        self.model.train() # Set model to training mode

        if self.verbose:
            print(f"Training MLP with hidden_size={self.hidden_size}, "
                  f"optimizer={self.optimizer_params['type']} (lr={self.optimizer.param_groups[0]['lr']}), "
                  f"loss={self.loss_params['type']}, epochs={self.epochs}, batch_size={self.batch_size}")

        for epoch in range(self.epochs):
            # Simple manual batching
            for i in range(0, len(X_tensor), self.batch_size):
                inputs = X_tensor[i:i+self.batch_size]
                labels = y_tensor[i:i+self.batch_size]

                
                outputs = self.model(inputs) # Forward pass
                loss = self.criterion(outputs, labels) # Compute loss

                # Backward and optimize
                self.optimizer.zero_grad()  # Reset gradients
                loss.backward()             # Backpropagate gradients
                self.optimizer.step()       # Update weights

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f'  Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
        return self

    def predict(self, X):
        """
        Predicts class labels for the input data.

        Args:
            X (np.array): Features to predict on.
        Returns:
            np.array: Predicted class labels.
        """

         # Convert sparse matrix to dense numpy array if necessary
        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        self.model.eval() # Set model to evaluation mode
        X_tensor = torch.tensor(X_dense, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted_encoded = torch.max(outputs.data, 1)

        # Decode numerical predictions back to original labels
        return self.label_encoder.inverse_transform(predicted_encoded.numpy())

    def predict_proba(self, X):
        """
        Predicts class probabilities for the input data.

        Args:
            X (np.array): Features to predict on.
        Returns:
            np.array: Predicted class probabilities.
        """

         # Convert sparse matrix to dense numpy array if necessary
        if issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        self.model.eval() # Set model to evaluation mode
        X_tensor = torch.tensor(X_dense, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.numpy()

