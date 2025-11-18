# src/train.py

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight # Added
import numpy as np # Added for sample_weight
from src.config import Config
# Removed load_data, preprocess_data, split_data as they are not directly used in this module's functions

def train_model(X_train, y_train, model_name='LogisticRegression', param_grid=None, cv_folds=5, use_randomized_search=False, n_iter=10):
    """
    Trains a specified machine learning model, optionally performing GridSearchCV with StratifiedKFold.
    Applies class weighting if Config.USE_CLASS_WEIGHTING is True.
    """
    model = None
    class_weights = None
    sample_weights = None

    if Config.USE_CLASS_WEIGHTING:
        classes = np.unique(y_train)
        class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, class_weights_array))
        print(f"Computed class weights: {class_weights}")
        
        # Create sample weights for models that need it during fit
        sample_weights = np.array([class_weights[label] for label in y_train])

    model_params = {}
    if model_name == 'LogisticRegression':
        model_params = Config.LOGISTIC_REGRESSION_PARAMS.copy()
        if Config.USE_CLASS_WEIGHTING:
            model_params['class_weight'] = class_weights
        model = LogisticRegression(**model_params)
    elif model_name == 'SVC':
        model_params = Config.SVC_PARAMS.copy()
        if Config.USE_CLASS_WEIGHTING:
            model_params['class_weight'] = class_weights
        model = SVC(**model_params)
    elif model_name == 'RandomForestClassifier':
        model_params = Config.RANDOM_FOREST_PARAMS.copy()
        if Config.USE_CLASS_WEIGHTING:
            model_params['class_weight'] = class_weights
        model = RandomForestClassifier(**model_params)
    elif model_name == 'XGBClassifier':
        model_params = Config.XGBOOST_PARAMS.copy()
        # XGBoost handles class weights via `sample_weight` in fit or `scale_pos_weight` for binary
        # For multi-class, `sample_weight` is generally preferred.
        model = XGBClassifier(**model_params)
        if Config.USE_CLASS_WEIGHTING and sample_weights is not None:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
    elif model_name == 'LGBMClassifier':
        model_params = Config.LIGHTGBM_PARAMS.copy()
        if Config.USE_CLASS_WEIGHTING:
            model_params['class_weight'] = class_weights # LGBMClassifier directly accepts class_weight dict for multi-class
        model = LGBMClassifier(**model_params)
    elif model_name == 'KNeighborsClassifier':
        model_params = Config.KNN_PARAMS.copy()
        model = KNeighborsClassifier(**model_params)
    elif model_name == 'DecisionTreeClassifier':
        model_params = Config.DECISION_TREE_PARAMS.copy()
        if Config.USE_CLASS_WEIGHTING:
            model_params['class_weight'] = class_weights
        model = DecisionTreeClassifier(**model_params)
    elif model_name == 'GaussianNB':
        model_params = Config.GAUSSIAN_NAIVE_BAYES_PARAMS.copy()
        model = GaussianNB(**model_params)
    elif model_name == 'AdaBoostClassifier':
        model_params = Config.ADABOOST_PARAMS.copy()
        model = AdaBoostClassifier(**model_params)
    elif model_name == 'GradientBoostingClassifier':
        model_params = Config.GRADIENT_BOOSTING_PARAMS.copy()
        model = GradientBoostingClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if param_grid:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=Config.RANDOM_STATE)
        
        if use_randomized_search:
            print(f"Performing RandomizedSearchCV for {model_name} with {n_iter} iterations and {cv_folds}-fold StratifiedKFold...")
            search_cv = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter,
                                           cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1,
                                           random_state=Config.RANDOM_STATE)
        else:
            print(f"Performing GridSearchCV for {model_name} with {cv_folds}-fold StratifiedKFold...")
            search_cv = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, 
                                   scoring='f1_macro', n_jobs=-1, verbose=1)
        
        # If using sample weights, pass them to search_cv's fit method,
        # but exclude KNeighborsClassifier which does not support sample_weight.
        if Config.USE_CLASS_WEIGHTING and sample_weights is not None and model_name != 'KNeighborsClassifier':
            search_cv.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            search_cv.fit(X_train, y_train)

        print(f"{'RandomizedSearchCV' if use_randomized_search else 'GridSearchCV'} complete for {model_name}.")
        print(f"Best parameters for {model_name}: {search_cv.best_params_}")
        print(f"Best macro F1 score for {model_name}: {search_cv.best_score_:.4f}")
        return search_cv.best_estimator_
    else:
        print(f"Training {model_name} model...")
        if Config.USE_CLASS_WEIGHTING and sample_weights is not None and model_name not in ['LogisticRegression', 'SVC', 'RandomForestClassifier', 'DecisionTreeClassifier', 'LGBMClassifier']:
            # For models that don't take class_weight in constructor but can take sample_weight in fit
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        print(f"{model_name} model trained successfully.")
        return model

def evaluate_model(model, X_val, y_val):
    """
    Evaluates the model using Macro F1 Score.
    """
    y_pred = model.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    return macro_f1

# Removed save_model function as it will be handled in main.py

if __name__ == '__main__':
    # This block is for testing purposes if run directly
    # In a full pipeline, main.py orchestrates these steps
    print("This script is intended to be imported and used by main.py.")
    print("No direct execution for full pipeline training here.")
