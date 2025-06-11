from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV # Import GridSearchCV and RandomizedSearchCV
from sklearn.pipeline import Pipeline # Useful for combining preprocessor and model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Import model-specific configurations and tuning parameters
from config import SKLEARN_MODEL_PARAMS, RANDOM_STATE, TUNING_PARAMS

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

    if model_type == 'svm':
        ModelClass = SVC
    elif model_type == 'random_forest':
        ModelClass = RandomForestClassifier
    elif model_type == 'decision_tree':
        ModelClass = DecisionTreeClassifier
    elif model_type == 'naive_bayes':
        ModelClass = MultinomialNB # Or GaussianNB depending on feature type
    elif model_type in ['cnn', 'rnn', 'lstm', 'transformer', 'llm']:
        raise NotImplementedError(f"Deep Learning and LLM models ({model_type}) require separate implementations "
                                  "using frameworks like TensorFlow/Keras and different data handling. "
                                  "This part of the code is a placeholder for future development.")
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                         "Please ensure 'MODEL_TYPE' in config.py is one of the supported types.")

    if perform_tuning:
        print(f"\nStarting Hyperparameter Tuning ({tuning_method.upper()} Search) for {model_type.upper()} model...")
        param_grid = TUNING_PARAMS.get(model_type)

        if not param_grid:
            raise ValueError(f"No tuning parameters defined for model type: {model_type} in TUNING_PARAMS.")

        estimator = ModelClass(**model_params) # Initialize with default or base params

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
        # Original cross-validation logic (without tuning)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1_scores = []

        for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
            X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            fold_model = ModelClass(**model_params)
            fold_model.fit(X_train_fold, y_train_fold)

            y_pred_fold = fold_model.predict(X_val_fold)

            fold_accuracies.append(accuracy_score(y_val_fold, y_pred_fold))
            fold_precisions.append(precision_score(y_val_fold, y_pred_fold))
            fold_recalls.append(recall_score(y_val_fold, y_pred_fold))
            fold_f1_scores.append(f1_score(y_val_fold, y_pred_fold))

            print(f"\n--- Fold {fold + 1} Metrics ({model_type.upper()}) ---")
            print(f"Accuracy: {fold_accuracies[-1]:.4f}")
            print(f"Precision: {fold_precisions[-1]:.4f}")
            print(f"Recall: {fold_recalls[-1]:.4f}")
            print(f"F1-Score: {fold_f1_scores[-1]:.4f}")

        print("\n--- Average Cross-Validation Metrics ---")
        print(f"Average Accuracy: {np.mean(fold_accuracies):.4f}")
        print(f"Average Precision: {np.mean(fold_precisions):.4f}")
        print(f"Average Recall: {np.mean(fold_recalls):.4f}")
        print(f"Average F1-Score: {np.mean(fold_f1_scores):.4f}")

        # Train the final model on the entire X_train dataset for deployment
        print(f"\nTraining final {model_type.upper()} model on the full training dataset...")
        final_model = ModelClass(**model_params)
        final_model.fit(X_train, y_train)
        print(f"Final {model_type.upper()} model training complete.")

    return final_model
