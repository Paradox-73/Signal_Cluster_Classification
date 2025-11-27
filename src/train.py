# src/train.py

import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, accuracy_score # Added accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split # Added train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src.config import Config

# Import the user's NN function (Assumes nn_model.py is moved to src folder)
try:
    from src.nn_model import create_nn_model
except ImportError:
    print("Warning: Could not import create_nn_model. Make sure nn_model.py is in the src folder.")

# --- Custom Wrapper for Keras to work with Scikit-Learn ---
class SklearnKerasWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, num_classes=None, 
                 layer_0_units=64, layer_1_units=32, layer_2_units=16,
                 layer_0_dropout=0.3, layer_1_dropout=0.3, layer_2_dropout=0.3,
                 optimizer='adam', epochs=50, batch_size=32, verbose=0):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.layer_0_units = layer_0_units
        self.layer_1_units = layer_1_units
        self.layer_2_units = layer_2_units
        self.layer_0_dropout = layer_0_dropout
        self.layer_1_dropout = layer_1_dropout
        self.layer_2_dropout = layer_2_dropout
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        if self.num_classes is None:
            self.num_classes = len(self.classes_)
            
        self.model = create_nn_model(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            layer_0_units=self.layer_0_units,
            layer_1_units=self.layer_1_units,
            layer_2_units=self.layer_2_units,
            layer_0_dropout=self.layer_0_dropout,
            layer_1_dropout=self.layer_1_dropout,
            layer_2_dropout=self.layer_2_dropout,
            optimizer=self.optimizer
        )
        
        # Simple handling of sample_weight if provided
        fit_args = {'epochs': self.epochs, 'batch_size': self.batch_size, 'verbose': self.verbose}
        if sample_weight is not None:
            fit_args['sample_weight'] = sample_weight
            
        self.model.fit(X, y, **fit_args)
        return self

    def predict(self, X):
        probas = self.model.predict(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X):
        return self.model.predict(X)

# --- Main Training Function ---

def train_model(X_train, y_train, model_name='LogisticRegression', param_grid=None, cv_folds=5, use_randomized_search=False, X_full_scaled=None, y_full_encoded=None):
    model = None
    class_weights = None
    sample_weights = None

    # Handle Class Weights
    if Config.USE_CLASS_WEIGHTING:
        classes = np.unique(y_train)
        class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, class_weights_array))
        sample_weights = np.array([class_weights[label] for label in y_train])

    # Initialize Models
    if model_name == 'LogisticRegression':
        model = LogisticRegression(**Config.LOGISTIC_REGRESSION_PARAMS)
    elif model_name == 'SVC':
        model = SVC(**Config.SVC_PARAMS)
    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(**Config.RANDOM_FOREST_PARAMS)
    elif model_name == 'XGBClassifier':
        model = XGBClassifier(**Config.XGBOOST_PARAMS)
    elif model_name == 'LGBMClassifier':
        model = LGBMClassifier(**Config.LIGHTGBM_PARAMS)
    elif model_name == 'KNeighborsClassifier':
        print("Using exact KNN implementation from data/knn.py...")
        
        # 1. Split training data for validation, exactly as in data/knn.py
        # This split is for hyperparameter tuning only
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_full_scaled, y_full_encoded, test_size=0.2, random_state=Config.RANDOM_STATE
        )

        # 2. Hyperparameter Tuning using GridSearchCV for n_neighbors 1 to 30
        knn_base = KNeighborsClassifier()
        param_grid_knn = {'n_neighbors': np.arange(1, 31)} # Exact range from data/knn.py

        grid_search = GridSearchCV(knn_base, param_grid_knn, cv=5, scoring='accuracy') # Removed n_jobs=-1
        grid_search.fit(X_train_split, y_train_split)

        print(f"Best K found for KNN: {grid_search.best_params_['n_neighbors']}")
        print(f"Best Cross-Validation Score (accuracy): {grid_search.best_score_:.4f}")
        
        # Evaluate on the internal validation set, as in data/knn.py
        best_knn = grid_search.best_estimator_
        val_predictions = best_knn.predict(X_val_split)
        print(f"Internal Validation Accuracy for KNN: {accuracy_score(y_val_split, val_predictions):.4f}")
        
        # 3. Retrain on the FULL X_full_scaled, y_full_encoded dataset with the best K
        # This matches data/knn.py's final training step for submission
        if X_full_scaled is not None and y_full_encoded is not None:
            model = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
            model.fit(X_full_scaled, y_full_encoded)
        else:
            print("Warning: X_full_scaled or y_full_encoded not provided. Training on X_train/y_train instead.")
            model = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
            model.fit(X_train, y_train)
        
        # Skip the generic hyperparameter tuning block later, as KNN is handled here
        param_grid = None # Indicate that tuning is already done for KNN
    elif model_name == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(**Config.DECISION_TREE_PARAMS)
    elif model_name == 'GaussianNB':
        model = GaussianNB(**Config.GAUSSIAN_NAIVE_BAYES_PARAMS)
    elif model_name == 'QDA':
        model = QuadraticDiscriminantAnalysis(**Config.QDA_PARAMS)
    elif model_name == 'AdaBoostClassifier':
        model = AdaBoostClassifier(**Config.ADABOOST_PARAMS)
    elif model_name == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(**Config.GRADIENT_BOOSTING_PARAMS)
    elif model_name == 'NeuralNetwork':
        # Pass input_dim explicitly
        model = SklearnKerasWrapper(
            input_dim=X_train.shape[1],
            num_classes=len(np.unique(y_train)),
            **Config.NN_PARAMS
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Prepare fit parameters (like sample_weights)
    fit_params = {}
    # KNN and NB don't support sample_weight in fit generally (or behave differently)
    if Config.USE_CLASS_WEIGHTING and sample_weights is not None:
        if model_name not in ['KNeighborsClassifier', 'GaussianNB', 'NeuralNetwork']:
            fit_params['sample_weight'] = sample_weights
        elif model_name == 'NeuralNetwork':
            # Our wrapper handles sample_weight
            fit_params['sample_weight'] = sample_weights

    # Hyperparameter Tuning
    if param_grid:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=Config.RANDOM_STATE)
        print(f"Performing RandomizedSearchCV for {model_name}...")
        
        search_cv = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_grid, 
            n_iter=Config.N_ITER_RANDOM_SEARCH,
            cv=cv, 
            scoring='f1_macro', 
            n_jobs=-1, # Parallel processing
            verbose=1,
            random_state=Config.RANDOM_STATE
        )
        
        search_cv.fit(X_train, y_train, **fit_params)
        print(f"Best parameters for {model_name}: {search_cv.best_params_}")
        return search_cv.best_estimator_
    else:
        print(f"Training {model_name} without tuning...")
        model.fit(X_train, y_train, **fit_params)
        return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    return f1_score(y_val, y_pred, average='macro')