from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV # Import GridSearchCV and RandomizedSearchCV
from sklearn.pipeline import Pipeline # Useful for combining preprocessor and model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier # <--- NEW IMPORT for scikit-learn MLP

from fcnn_wrapper import PyTorchMLPClassifier 

import numpy as np

# Import model-specific configurations and tuning parameters
from config import SKLEARN_MODEL_PARAMS, RANDOM_STATE, TUNING_PARAMS, LOSS_PARAMS, OPTIMIZER_PARAMS

def get_model(model_type, model_params, random_state=None):
    """
    Returns an initialized model based on the specified type and parameters.
    """
    ModelClass = None # Initialize ModelClass

    if model_type == 'svm':
        ModelClass = SVC
    elif model_type == 'random_forest':
        ModelClass = RandomForestClassifier
    elif model_type == 'decision_tree':
        ModelClass = DecisionTreeClassifier
    elif model_type == 'naive_bayes':
        # Assuming MultinomialNB for text-based features, adjust if using GaussianNB
        ModelClass = MultinomialNB
    elif model_type == 'mlp': 
        ModelClass = MLPClassifier
    elif model_type == 'fcnn':
        ModelClass = PyTorchMLPClassifier
    else:
        raise ValueError(f"Unknown model type: {model_type}")


    if model_type == 'fcnn':
        model = ModelClass(
            hidden_size=model_params.get('hidden_size', 64),
            learning_rate=model_params.get('learning_rate', 0.001),
            epochs=model_params.get('epochs', 50),
            batch_size=model_params.get('batch_size', 32),
            random_state=random_state, 
            verbose=False,
             optimizer_params=OPTIMIZER_PARAMS, 
             loss_params=LOSS_PARAMS 
        )
    else:
        if 'random_state' in ModelClass()._get_param_names(): 
            model = ModelClass(random_state=random_state, **model_params)
        else:
            model = ModelClass(**model_params)

    return model

def train_model(X_train, y_train, model_type, n_splits=5, random_state=RANDOM_STATE, perform_tuning=False, tuning_method='grid', random_search_n_iter=10):
    """
    Trains a machine learning model using Stratified K-Fold Cross-Validation.
    Optionally performs hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

    Args:
        X_train (array-like): Training features.
        y_train (pandas.Series): Training labels.
        model_type (str): Type of model to train (e.g., 'svm', 'random_forest').
        n_splits (int): Number of folds for cross-validation.
        random_state (int): Random seed for reproducibility.
        perform_tuning (bool): If True, performs hyperparameter tuning.
        tuning_method (str): 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
        random_search_n_iter (int): Number of parameter settings that are sampled if using RandomizedSearchCV.

    Returns:
        object: The trained final model (or the best estimator from tuning).
    """
    print(f"\nPerforming {n_splits}-fold Stratified Cross-Validation on training data for {model_type.upper()} model...")

    # Dynamically select model and its parameters
    model_params = SKLEARN_MODEL_PARAMS.get(model_type, {})
    if 'random_state' in model_params and model_params['random_state'] is None:
        model_params['random_state'] = random_state

    model = get_model(model_type, model_params, random_state)

    if perform_tuning:
        print(f"\nStarting Hyperparameter Tuning ({tuning_method.upper()} Search) for {model_type.upper()} model...")
        param_grid = TUNING_PARAMS.get(model_type)

        if not param_grid:
            raise ValueError(f"No tuning parameters defined for model type: {model_type} in TUNING_PARAMS.")

        estimator = model

        # Define the cross-validation strategy for tuning
        cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        if tuning_method == 'grid':
            search_cv = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=cv_strategy,
                scoring='f1', # Optimize for F1-score, common for imbalanced datasets
                n_jobs=-1, # Use all available CPU cores
                verbose=2
            )
        elif tuning_method == 'random':
            search_cv = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid, # For RandomizedSearchCV, use param_distributions
                n_iter=random_search_n_iter, # Number of parameter settings that are sampled
                cv=cv_strategy,
                scoring='f1',
                n_jobs=-1,
                verbose=2,
                random_state=random_state # For reproducibility of random search
            )
        else:
            raise ValueError(f"Unsupported tuning method: {tuning_method}. Must be 'grid' or 'random'.")

        search_cv.fit(X_train, y_train)

        print("\nHyperparameter Tuning Complete.")
        print(f"Best parameters for {model_type.upper()}: {search_cv.best_params_}")
        print(f"Best cross-validation F1-score: {search_cv.best_score_:.4f}")

        final_model = search_cv.best_estimator_ # The model with the best parameters
    else:
        # No cross validation, nor hyperparameter tuning
        final_model = model
        model.fit(X_train, y_train)
        print(f"{model_type.upper()} model training complete.")

    return final_model
