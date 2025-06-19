from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline # Useful for combining preprocessor and model
from sklearn.neural_network import MLPClassifier # <--- NEW IMPORT for scikit-learn MLP

from MLP_wrapper import PyTorchMLPClassifier 
from CNN_wrapper import CNNClassifier

import numpy as np

from config import (
    MODEL_TYPE,
    N_SPLITS_CROSS_VALIDATION,
    RANDOM_STATE,
    PERFORM_TUNING,
    TUNING_METHOD,
    RANDOM_SEARCH_N_ITER, MODEL_PARAMS, TUNING_PARAMS
)

def get_model(preprocessor):
    """
    Returns an initialized model based on the specified type and parameters.
    """

    model_type = MODEL_TYPE.lower()
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
    ## deep learning models
    elif model_type == 'fcnn':
        ModelClass = PyTorchMLPClassifier
    elif model_type == 'cnn':
        ModelClass = CNNClassifier
    else:
        raise ValueError(f"Unknown model type: {model_type}")



    # Dynamically select model and its parameters
    model_params = MODEL_PARAMS.get(MODEL_TYPE, {})
    
    if  model_type == 'mlp'or  model_type == 'cnn':
        model_params['preprocessor'] = preprocessor

    if 'random_state' in model_params and model_params['random_state'] is None:
        model_params['random_state'] = RANDOM_STATE

    model = ModelClass(**model_params)

    return model

def train_model(X_train, y_train, preprocessor, logger):
    """
    Trains a machine learning model using Stratified K-Fold Cross-Validation.
    Optionally performs hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

    Args:
        X_train (array-like): Training features.
        y_train (pandas.Series): Training labels.
      
    Returns:
        object: The trained final model (or the best estimator from tuning).
    """
    logger.info("--- Model Training ---")
    log_message = f"""\tAlgorithm Used: {MODEL_TYPE.upper()}
\tModel Parameters: {MODEL_PARAMS[MODEL_TYPE]}
\tHyperparameter tuning: {PERFORM_TUNING}
\tHyperparameter tuning method: {TUNING_METHOD}
\tCross-Validation Splits: {N_SPLITS_CROSS_VALIDATION}
\tNum of hyperparameter combinations in Random Search: {RANDOM_SEARCH_N_ITER}
\tRandom State: {RANDOM_STATE}
\tTuning Parameters: {TUNING_PARAMS[MODEL_TYPE]}"""
    logger.info(log_message) 
    
    print(f"\nPerforming {N_SPLITS_CROSS_VALIDATION}-fold Stratified Cross-Validation on training data for {MODEL_TYPE.upper()} model...")

   

    model = get_model(preprocessor)

    if PERFORM_TUNING:
        print(f"\nStarting Hyperparameter Tuning ({TUNING_METHOD.upper()} Search) for {MODEL_TYPE.upper()} model...")
        param_grid = TUNING_PARAMS.get(MODEL_TYPE)

        if not param_grid:
            raise ValueError(f"No tuning parameters defined for model type: {MODEL_TYPE} in TUNING_PARAMS.")

        estimator = model

        # Define the cross-validation strategy for tuning
        cv_strategy = StratifiedKFold(n_splits=N_SPLITS_CROSS_VALIDATION, shuffle=True, random_state=RANDOM_STATE)

        if TUNING_METHOD == 'grid':
            search_cv = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=cv_strategy,
                scoring='f1', # Optimize for F1-score, common for imbalanced datasets
                n_jobs=-1, # Use all available CPU cores
                verbose=2
            )
        elif TUNING_METHOD == 'random':
            search_cv = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid, # For RandomizedSearchCV, use param_distributions
                n_iter=RANDOM_SEARCH_N_ITER, # Number of parameter settings that are sampled
                cv=cv_strategy,
                scoring='f1',
                n_jobs=-1,
                verbose=2,
                random_state=RANDOM_STATE # For reproducibility of random search
            )
        else:
            raise ValueError(f"Unsupported tuning method: {TUNING_METHOD}. Must be 'grid' or 'random'.")

        search_cv.fit(X_train, y_train)

        print("\nHyperparameter Tuning Complete.")
        print(f"Best parameters for {MODEL_TYPE.upper()}: {search_cv.best_params_}")
        print(f"Best cross-validation F1-score: {search_cv.best_score_:.4f}")

        final_model = search_cv.best_estimator_ # The model with the best parameters
    else:
        # No cross validation, nor hyperparameter tuning
        final_model = model
        final_model.fit(X_train, y_train)
        print(f"{MODEL_TYPE.upper()} model training complete.")

    if MODEL_TYPE.lower() == 'fcnn' or MODEL_TYPE.lower() == 'cnn' or MODEL_TYPE.lower() == 'rnn'  or MODEL_TYPE.lower() == 'lstm':
        print('####################################################')
        print(final_model.model)
    return final_model
