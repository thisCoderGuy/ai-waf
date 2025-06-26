import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder 
from scipy.sparse import issparse
from torch.utils.data import TensorDataset, DataLoader

from loggers import global_logger, evaluation_logger

from config import (
    TEXT_FEATURES, CATEGORICAL_FEATURES,  NUMERICAL_FEATURES
)
from http_request_dataset import HTTPRequestMultiInputDataset

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
                 random_state=None,
                 optimizer_type='adam',  optimizer_params=None,
                 loss_type='CrossEntropyLoss',  loss_params=None):
        """
        Initializes the base PyTorch classifier.

        Args:
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
            batch_size (int): Size of training mini-batches.
            random_state (int): Seed for reproducibility.
            optimizer_params (dict): Dictionary with 'type' and 'hyperparameters' for the optimizer.
                                     If None, defaults from config.py will be used.
            loss_params (dict): Dictionary with 'type' and 'hyperparameters' for the loss function.
                                If None, defaults from config.py will be used.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
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
        evaluation_logger.info(f"Device: {self.device}")

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
            X (np.array or sparse matrix): Preprocessed training features.
            y (np.array): Training labels.
        Returns:
            self: The trained classifier.
        """
        
        # Build the model components if not already built
        if self.model is None:
            self._build_model_components()

        # Create DataLoader for batching
        # Combine everything into one dataset
        train_dataset = HTTPRequestMultiInputDataset(
            X=X, 
            y=y
        )
     
        global_logger.debug(f"{len(train_dataset)=}")
        #for idx, sample in enumerate(train_dataset): # Sample = row. Each sample is a tuple of tensors (One tensor for each feature)
        #    for i, part in enumerate(sample):  # PArt = a tensor for each of the features of a sample (row)
        #for part in train_dataset[0]:            
        #    global_logger.debug(f" Sample 0 parts: {type(part)=} ")
            #global_logger.debug(f" Sample 0 parts: {part.shape=} {part.dtype=} {part.ndim=} {part.device.type=}")
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        global_logger.info(f"Starting training:")
        evaluation_logger.info(f"  Model type: {self.__class__.__name__}")
        evaluation_logger.info(f"  Optimizer: {self.optimizer_type} ({self.optimizer_params})")
        evaluation_logger.info(f"  Loss: {self.loss_type} ({self.loss_params})")
        evaluation_logger.info(f"  Epochs: {self.epochs}, Batch Size: {self.batch_size}")
        evaluation_logger.info(f"  Device: {self.device}")

        # TODO
        # L1 regularization
        # l1_lambda = 0.001
        # L2 regularization , same as weight_decay param in Adam Optimizer
        # l2_lambda = 0.001

        for epoch in range(self.epochs):
            self.model.train() # Set model to training mode
            total_loss = 0

            for i, batch in enumerate(train_loader):
                
                #for i, part in enumerate(batch):
                #    global_logger.debug(f"Batch part: {part.shape=} {part.dtype=} {part.ndim=} {part.device.type=}")
    
                # Unpack the batch into individual components
                # The order here must match the return order from CustomMultiInputDataset's __getitem__
                batch_text_tensors, batch_cat_tensors, batch_num_tensors, batch_labels = batch

                # Ensure all tensors in the lists are moved to device
                batch_text_tensors = [t.to(self.device) for t in batch_text_tensors]
                batch_cat_tensors = [t.to(self.device) for t in batch_cat_tensors]
                batch_num_tensors = [t.to(self.device) for t in batch_num_tensors]
                batch_labels = batch_labels.to(self.device)

                global_logger.debug(f"--- Batch {i} for epoch {epoch}---")
                global_logger.debug(f"batch_text_tensors: {type(batch_text_tensors)=}")
                for i, batch_text_tensor in enumerate(batch_text_tensors):
                    global_logger.debug(f"     batch_text_tensor {TEXT_FEATURES[i]}: {batch_text_tensor.shape=} {batch_text_tensor.dtype=} {batch_text_tensor.ndim=} {batch_text_tensor.device.type=}")
            
              
                global_logger.debug(f"batch_cat_tensors: {type(batch_cat_tensors)=}")
                for i, batch_cat_tensor in enumerate(batch_cat_tensors):
                    global_logger.debug(f"     batch_cat_tensor {CATEGORICAL_FEATURES[i]}: {batch_cat_tensor.shape=} {batch_cat_tensor.dtype=} {batch_cat_tensor.ndim=} {batch_cat_tensor.device.type=}")
                
                global_logger.debug(f"batch_num_tensors: {type(batch_num_tensors)=}")
                for i, batch_num_tensor in enumerate(batch_num_tensors):
                    global_logger.debug(f"     batch_num_tensor {NUMERICAL_FEATURES[i]}: {batch_num_tensor.shape=} {batch_num_tensor.dtype=} {batch_num_tensor.ndim=} {batch_num_tensor.device.type=}")
                
               
                global_logger.debug(f"batch labels: {batch_labels.shape=} {batch_labels.dtype=} {batch_labels.ndim=} {batch_labels.device.type=}")
                
                # Forward pass
                logits = self.model( 
                    text_inputs=batch_text_tensors, #  dictionary of tensors, one per text feature (each of shape [batch_size, seq_len])
                    categorical_inputs=batch_cat_tensors, # dictionary of tensors, one per categorical feature (each of shape [batch_size, 1])
                    numerical_inputs=batch_num_tensors # dictionary of tensors, one per categorical feature (each of shape [batch_size, 1])
                ) #.squeeze()  # For binary classification
                # logits shape: (batch_size, num_classes) 

                loss = self.criterion(logits, batch_labels.long())  # Use CrossEntropyLoss
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
            global_logger.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

        
        return self

    def predict(self, X):
        """
        Predicts the class labels for the given input features.

        Args:
            X (np.array or sparse matrix): Input features for prediction.

        Returns:
            np.array: Predicted class labels (0 or 1 for binary classification).
        """
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)


    def predict_proba(self, X):
        """
        Predicts the class probabilities for the given input features.

        Args:
            X (np.array or sparse matrix): Input features for probability prediction.

        Returns:
            np.array: Predicted class probabilities (e.g., shape (n_samples, n_classes)).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call .fit() first.")

        self.model.eval() # Set model to evaluation mode

        dataset = HTTPRequestMultiInputDataset(
            X=X, 
            y=None
        )

        loader = DataLoader(dataset, batch_size=self.batch_size)

        all_probs = []

        with torch.no_grad():
            for batch in loader:
                batch_text_tensors, batch_cat_tensors, batch_num_tensors = batch

                # Ensure all tensors in the lists are moved to device
                batch_text_tensors = [t.to(self.device) for t in batch_text_tensors]
                batch_cat_tensors = [t.to(self.device) for t in batch_cat_tensors]
                batch_num_tensors = [t.to(self.device) for t in batch_num_tensors]

                # Forward pass
                logits = self.model(
                    text_inputs=batch_text_tensors,
                    categorical_inputs=batch_cat_tensors,
                    numerical_inputs=batch_num_tensors
                )

                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu())

        return torch.cat(all_probs, dim=0).numpy()