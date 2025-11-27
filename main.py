# main.py

import pandas as pd
import joblib
import os
import sys
import numpy as np
from src.config import Config
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.train import train_model, evaluate_model
from src.predict import generate_predictions, create_submission_file
from src.eda import visualize_model_decision_boundaries, visualize_clusters
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from src.ensemble import create_voting_ensemble, create_stacking_ensemble
from sklearn.linear_model import LogisticRegression

def process_single_model(model_name, X_train, y_train, X_val, y_val, X_test_scaled, label_encoder, test_df, X_full_scaled, y_full_encoded):
    """Helper to train, evaluate, and predict for a single model."""
    print(f"\n--- Processing {model_name} ---")
    
    # 1. Get Grid
    param_grid = None
    if model_name == 'LogisticRegression': param_grid = Config.LOGISTIC_REGRESSION_GRID
    elif model_name == 'SVC': param_grid = Config.SVC_GRID
    elif model_name == 'RandomForestClassifier': param_grid = Config.RANDOM_FOREST_GRID
    elif model_name == 'XGBClassifier': param_grid = Config.XGBOOST_GRID
    elif model_name == 'LGBMClassifier': param_grid = Config.LIGHTGBM_GRID
    # Removed: elif model_name == 'KNeighborsClassifier': param_grid = Config.KNN_GRID
    elif model_name == 'DecisionTreeClassifier': param_grid = Config.DECISION_TREE_GRID
    elif model_name == 'AdaBoostClassifier': param_grid = Config.ADABOOST_GRID
    elif model_name == 'GradientBoostingClassifier': param_grid = Config.GRADIENT_BOOSTING_GRID
    elif model_name == 'NeuralNetwork': param_grid = Config.NN_GRID

    # 2. Train
    model = train_model(X_train, y_train, model_name=model_name, param_grid=param_grid, X_full_scaled=X_full_scaled, y_full_encoded=y_full_encoded)
    
    # 3. Evaluate
    score = evaluate_model(model, X_val, y_val)
    print(f"Validation Macro F1 Score for {model_name}: {score:.4f}")

    # 4. Save
    model_path = os.path.join(Config.MODELS_DIR, f"{model_name.lower()}_model.pkl")
    joblib.dump(model, model_path)

    # 5. Predict & Submit
    predictions = generate_predictions(model, X_test_scaled, label_encoder)
    create_submission_file(test_df, predictions, model_name)
    
    # Visualize clusters for KNeighborsClassifier
    if model_name == 'KNeighborsClassifier':
        # X_scaled from preprocess_data is the full dataset, which is what we want to visualize
        visualize_clusters(model, X_full_scaled, y_full_encoded, model_name, label_encoder, method='pca')
    
    return model, score

def _load_or_get_model(model_name, trained_models, X_val, y_val, X_test_scaled, label_encoder, test_df):
    """
    Helper function to load a model from disk if available, otherwise fetch from trained_models.
    If loaded from disk, it also evaluates the model and updates trained_models.
    """
    if model_name in trained_models:
        return trained_models[model_name]
    
    model_path = os.path.join(Config.MODELS_DIR, f"{model_name.lower()}_model.pkl")
    if os.path.exists(model_path):
        print(f"Loading pre-trained {model_name} model for ensemble from {model_path}")
        model = joblib.load(model_path)
        # Evaluate to get a score for the loaded model if needed, or just return
        # For ensemble purposes, we primarily need the model object.
        # However, for consistency with `process_single_model`, we can also evaluate.
        # score = evaluate_model(model, X_val, y_val)
        # print(f"Validation Macro F1 Score for loaded {model_name}: {score:.4f}")
        trained_models[model_name] = model # Add to trained_models for future use
        return model
    else:
        print(f"Warning: Model {model_name} not found in trained_models or on disk. It will be skipped for ensemble.")
        return None

def run_knn_svm_stacking(trained_models, X_train, y_train, X_val, y_val, X_test_scaled, label_encoder, test_df):
    """Runs the specialized KNN + SVM Stacking Ensemble."""
    print("\n--- Processing Specialized KNN + SVM Stacking Ensemble ---")
    
    required = ['KNeighborsClassifier', 'SVC']
    required_models = {}
    for name in required:
        model = _load_or_get_model(name, trained_models, X_val, y_val, X_test_scaled, label_encoder, test_df)
        if model:
            required_models[name] = model

    if len(required_models) < 2:
        print("Error: Both KNN and SVC must be available (trained or saved) for stacking.")
        return 0.0

    knn_svc_stack = create_stacking_ensemble(
        required_models, 
        meta_model=LogisticRegression(C=1.0, random_state=Config.RANDOM_STATE)
    )
    
    knn_svc_stack.fit(X_train, y_train)
    score = evaluate_model(knn_svc_stack, X_val, y_val)
    print(f"Validation Macro F1 Score for KNN + SVM Stack: {score:.4f}")
    
    joblib.dump(knn_svc_stack, os.path.join(Config.MODELS_DIR, "knn_svm_stacking_model.pkl"))
    preds = generate_predictions(knn_svc_stack, X_test_scaled, label_encoder)
    create_submission_file(test_df, preds, "KNN_SVM_Stacking")
    return score

def main():
    print("Starting SignalCluster Classification Pipeline...")

    # --- Data Loading & Preprocessing (Always run this) ---
    print("Loading and Preprocessing data...")
    train_df, test_df = load_data()
    X_scaled, y_encoded, X_test_scaled, label_encoder, scaler = preprocess_data(
        train_df, test_df, 
        use_polynomial_features=Config.USE_POLYNOMIAL_FEATURES, 
        polynomial_degree=Config.POLYNOMIAL_DEGREE,
        use_additional_engineered_features=Config.USE_ADDITIONAL_ENGINEERED_FEATURES
    )
    X_train, X_val, y_train, y_val = split_data(X_scaled, y_encoded)
    
    feature_names = X_scaled.columns.tolist()
    class_names = label_encoder.inverse_transform(sorted(np.unique(y_encoded)))
    
    trained_models = {}
    results = {}
    
    # --- Menu System ---
    available_models = [
        'LogisticRegression', 'SVC', 'RandomForestClassifier', 'XGBClassifier', 
        'LGBMClassifier', 'KNeighborsClassifier', 'DecisionTreeClassifier', 
        'GaussianNB', 'QDA', 'AdaBoostClassifier', 'GradientBoostingClassifier',
        'NeuralNetwork'
    ]

    while True:
        print("\n" + "="*40)
        print("   SIGNAL CLASSIFICATION MENU")
        print("="*40)
        print("1. Train Specific Model")
        print("2. Train ALL Base Models")
        print("3. Train Specialized KNN + SVM Stack (Requires KNN & SVC)")
        print("4. Train Voting Ensemble (Requires top models)")
        print("5. Train Stacking Ensemble (Requires top models)")
        print("6. Show Results & Exit")
        print("0. Exit")
        
        choice = input("\nEnter choice: ")
        
        if choice == '1':
            print("\nAvailable Models:")
            for i, m in enumerate(available_models):
                print(f"{i+1}. {m}")
            try:
                m_idx = int(input("Select model number: ")) - 1
                if 0 <= m_idx < len(available_models):
                    m_name = available_models[m_idx]
                    model, score = process_single_model(
                        m_name, X_train, y_train, X_val, y_val, 
                        X_test_scaled, label_encoder, test_df,
                        X_scaled, y_encoded
                    )
                    trained_models[m_name] = model
                    results[m_name] = score
                else:
                    print("Invalid model selection.")
            except ValueError:
                print("Invalid input.")

        elif choice == '2':
            for m_name in available_models:
                try:
                    model, score = process_single_model(
                        m_name, X_train, y_train, X_val, y_val, 
                        X_test_scaled, label_encoder, test_df
                    )
                    trained_models[m_name] = model
                    results[m_name] = score
                except Exception as e:
                    print(f"Failed to train {m_name}: {e}")

        elif choice == '3':
            score = run_knn_svm_stacking(trained_models, X_train, y_train, X_val, y_val, X_test_scaled, label_encoder, test_df)
            results['KNN_SVM_Stack'] = score

        elif choice == '4': # Voting Ensemble
            print("\n--- Processing Voting Ensemble ---")
            top_model_names = ['SVC', 'XGBClassifier', 'KNeighborsClassifier', 'LGBMClassifier', 'RandomForestClassifier']
            
            ensemble_estimators = []
            for name in top_model_names:
                model = _load_or_get_model(name, trained_models, X_val, y_val, X_test_scaled, label_encoder, test_df)
                if model:
                    ensemble_estimators.append((name, model))

            if not ensemble_estimators:
                print("No suitable models found for Voting.")
            else:
                base_voting_clf = create_voting_ensemble(ensemble_estimators)
                # Quick tune
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)
                rs = RandomizedSearchCV(base_voting_clf, Config.VOTING_CLASSIFIER_GRID, cv=cv, scoring='f1_macro', n_iter=10, n_jobs=-1, verbose=1)
                rs.fit(X_train, y_train)
                
                voting_model = rs.best_estimator_
                score = evaluate_model(voting_model, X_val, y_val)
                print(f"Voting Ensemble Score: {score:.4f}")
                results['VotingClassifier'] = score
                
                joblib.dump(voting_model, os.path.join(Config.MODELS_DIR, "voting_classifier.pkl"))
                preds = generate_predictions(voting_model, X_test_scaled, label_encoder)
                create_submission_file(test_df, preds, "VotingClassifier")

        elif choice == '5': # Stacking Ensemble
            print("\n--- Processing Stacking Ensemble ---")
            top_model_names = ['SVC', 'XGBClassifier', 'KNeighborsClassifier', 'LGBMClassifier', 'RandomForestClassifier']
            stacking_base = {}
            for name in top_model_names:
                model = _load_or_get_model(name, trained_models, X_val, y_val, X_test_scaled, label_encoder, test_df)
                if model:
                    stacking_base[name] = model
            
            if not stacking_base:
                print("No suitable models found for Stacking.")
            else:
                stacking_clf = create_stacking_ensemble(stacking_base)
                stacking_clf.fit(X_train, y_train)
                score = evaluate_model(stacking_clf, X_val, y_val)
                print(f"Stacking Ensemble Score: {score:.4f}")
                results['StackingClassifier'] = score
                
                joblib.dump(stacking_clf, os.path.join(Config.MODELS_DIR, "stacking_classifier.pkl"))
                preds = generate_predictions(stacking_clf, X_test_scaled, label_encoder)
                create_submission_file(test_df, preds, "StackingClassifier")

        elif choice == '6':
            print("\n--- Final Results Summary ---")
            for model_name, score in results.items():
                print(f"- {model_name}: {score:.5f}")
            break
            
        elif choice == '0':
            print("Exiting.")
            sys.exit()
        else:
            print("Invalid choice.")

if __name__ == '__main__':
    main()