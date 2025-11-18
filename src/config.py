# src/config.py

import os

class Config:
    # Data paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
    TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
    TEST_FILE = os.path.join(DATA_DIR, 'test.csv')
    SAMPLE_SUBMISSION_FILE = os.path.join(DATA_DIR, 'sample_submission.csv')

    # Model paths
    MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
    # BEST_MODEL_PATH will be dynamically generated based on model name
    # SUBMISSION_FILE will be dynamically generated based on model name

    # Model parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_ITER_RANDOM_SEARCH = 200 # Default number of iterations for RandomizedSearchCV

    # Hyperparameters for different models
    LOGISTIC_REGRESSION_PARAMS = {
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    }
    SVC_PARAMS = {
        'kernel': 'rbf',
        'random_state': RANDOM_STATE,
        'probability': True # Needed for consistent predict_proba if used later
    }
    RANDOM_FOREST_PARAMS = {
        'n_estimators': 100,
        'random_state': RANDOM_STATE
    }
    XGBOOST_PARAMS = {
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'use_label_encoder': False, # Suppress warning
        'num_class': 3, # Assuming 3 classes (Group_A, Group_B, Group_C)
        'random_state': RANDOM_STATE,
        'verbosity': 0
    }
    LIGHTGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': 3, # Assuming 3 classes
        'random_state': RANDOM_STATE,
        'verbose': -1 # Suppress LightGBM output
    }
    KNN_PARAMS = {
        'n_neighbors': 5
    }
    DECISION_TREE_PARAMS = {
        'random_state': RANDOM_STATE
    }
    GAUSSIAN_NAIVE_BAYES_PARAMS = {} # No common parameters for basic usage
    ADABOOST_PARAMS = {
        'n_estimators': 50,
        'random_state': RANDOM_STATE
    }
    GRADIENT_BOOSTING_PARAMS = {
        'n_estimators': 100,
        'random_state': RANDOM_STATE
    }

    # Hyperparameter Grids for GridSearchCV
    SVC_GRID = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1]
    }

    RANDOM_FOREST_GRID = {
        'n_estimators': [100, 200, 300, 500, 800, 1000],
        'max_depth': [5, 10, 20, 30, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8]
    }

    XGBOOST_GRID = {
        'n_estimators': [100, 200, 300, 500, 800],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 9, 12]
    }

    LIGHTGBM_GRID = {
        'n_estimators': [100, 200, 300, 500, 800],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'num_leaves': [20, 31, 40, 50, 60, 80],
        'verbose': [-1]
    }

    KNN_GRID = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }

    DECISION_TREE_GRID = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    ADABOOST_GRID = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }

    GRADIENT_BOOSTING_GRID = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    LOGISTIC_REGRESSION_GRID = {
        'C': [0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'saga'],
        'max_iter': [500, 1000, 2000]
    }

    # Feature Engineering
    USE_POLYNOMIAL_FEATURES = True
    POLYNOMIAL_DEGREE = 2
    USE_ADDITIONAL_ENGINEERED_FEATURES = False # New flag to enable/disable additional engineered features
    # Class Imbalance Techniques
    USE_CLASS_WEIGHTING = True

    # VotingClassifier Grid for optimized weights (for top 5 models: SVC, XGB, KNN, LGBM, RF)
    VOTING_CLASSIFIER_GRID = {
        'weights': [
            [0.2, 0.2, 0.2, 0.2, 0.2],  # Equal weights
            [0.3, 0.2, 0.2, 0.15, 0.15], # Biased towards top performers (SVC, XGB)
            [0.4, 0.2, 0.2, 0.1, 0.1],   # More biased towards top performer (SVC)
            [0.1, 0.1, 0.4, 0.2, 0.2],   # Biased towards KNN
            [0.25, 0.25, 0.25, 0.125, 0.125], # Slightly biased towards top 3
            [0.1, 0.2, 0.3, 0.2, 0.2], # Varying weights
            [0.2, 0.1, 0.2, 0.3, 0.2],
            [0.2, 0.2, 0.1, 0.2, 0.3],
            [0.3, 0.1, 0.1, 0.3, 0.2],
            [0.1, 0.3, 0.1, 0.2, 0.3],
            [0.05, 0.05, 0.4, 0.25, 0.25], # Strong bias towards KNN
            [0.3, 0.3, 0.1, 0.1, 0.2], # Strong bias towards SVC, XGB
            [0.1, 0.1, 0.1, 0.35, 0.35] # Strong bias towards LGBM, RF
        ]
    }
