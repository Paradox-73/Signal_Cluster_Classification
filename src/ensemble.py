# src/ensemble.py

import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from src.config import Config

def create_voting_ensemble(estimators, weights=None):
    """
    Creates a VotingClassifier ensemble from a list of (name, model) tuples.
    """
    voting_clf = VotingClassifier(estimators=estimators, voting='soft', weights=weights, n_jobs=-1)
    return voting_clf

def create_stacking_ensemble(base_models, meta_model=None, cv=5):
    """
    Creates a StackingClassifier ensemble from a dictionary of base models.
    """
    estimators = [(name, model) for name, model in base_models.items()]
    if meta_model is None:
        meta_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=Config.RANDOM_STATE)
    
    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=cv, n_jobs=-1)
    return stacking_clf
