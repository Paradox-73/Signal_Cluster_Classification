# main.py

import pandas as pd
import joblib
import os
import numpy as np # Added
from src.config import Config
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.train import train_model, evaluate_model
from src.predict import generate_predictions, create_submission_file
from src.eda import visualize_decision_boundary
from sklearn.metrics import f1_score # Added for direct use if needed
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from src.ensemble import create_voting_ensemble, create_stacking_ensemble # Added

def main():
    print("Starting SignalCluster Classification pipeline...")

    # 1. Load Data
    print("Loading data...")
    train_df, test_df = load_data()
    print("Data loaded successfully.")

    # 2. Preprocess Data
    print("Preprocessing data...")
    X_scaled, y_encoded, X_test_scaled, label_encoder, scaler = preprocess_data(
        train_df, test_df, 
        use_polynomial_features=Config.USE_POLYNOMIAL_FEATURES, 
        polynomial_degree=Config.POLYNOMIAL_DEGREE
    )
    print("Data preprocessed successfully.")

    # 3. Split Data
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = split_data(X_scaled, y_encoded)
    print("Data split successfully.")

    # Get feature names and class names for visualization
    feature_names = X_scaled.columns.tolist()
    class_names = label_encoder.inverse_transform(sorted(np.unique(y_encoded))) # Fixed


    # Define models to train
    models_to_train = [
        'LogisticRegression',
        'SVC',
        'RandomForestClassifier',
        'XGBClassifier',
        'LGBMClassifier',
        'KNeighborsClassifier',
        'DecisionTreeClassifier',
        'GaussianNB',
        'AdaBoostClassifier',
        'GradientBoostingClassifier'
    ]

    results = {}
    trained_models = {} # To store models for ensembling

    for model_name in models_to_train:
        print(f"\n--- Processing {model_name} ---")

        param_grid = None
        if model_name == 'LogisticRegression':
            param_grid = Config.LOGISTIC_REGRESSION_GRID # Assuming a grid will be added for LR if needed
        elif model_name == 'SVC':
            param_grid = Config.SVC_GRID
        elif model_name == 'RandomForestClassifier':
            param_grid = Config.RANDOM_FOREST_GRID
        elif model_name == 'XGBClassifier':
            param_grid = Config.XGBOOST_GRID
        elif model_name == 'LGBMClassifier':
            param_grid = Config.LIGHTGBM_GRID
        elif model_name == 'KNeighborsClassifier':
            param_grid = Config.KNN_GRID
        elif model_name == 'DecisionTreeClassifier':
            param_grid = Config.DECISION_TREE_GRID
        elif model_name == 'AdaBoostClassifier':
            param_grid = Config.ADABOOST_GRID
        elif model_name == 'GradientBoostingClassifier':
            param_grid = Config.GRADIENT_BOOSTING_GRID
        
        # 4. Train Model
        model = train_model(X_train, y_train, model_name=model_name, param_grid=param_grid, use_randomized_search=True, n_iter=50)
        trained_models[model_name] = model # Store the trained model

        # 5. Evaluate Model
        print(f"Evaluating {model_name} on validation set...")
        macro_f1 = evaluate_model(model, X_val, y_val)
        print(f"Validation Macro F1 Score for {model_name}: {macro_f1}")
        results[model_name] = macro_f1

        # 6. Visualize Decision Boundary
        # Only visualize if the number of features is 2 (i.e., no polynomial features applied)
        if X_train.shape[1] == 2:
            print(f"Visualizing decision boundary for {model_name}...")
            visualize_decision_boundary(model, X_train, y_train, 
                                        feature_names, class_names, 
                                        title=f'Decision Boundary of {model_name}',
                                        filename=f'decision_boundary_{model_name.lower()}.png')
        else:
            print(f"Skipping decision boundary visualization for {model_name} as feature count is {X_train.shape[1]} (not 2).")


        # 7. Save Model
        model_path = os.path.join(Config.MODELS_DIR, f"{model_name.lower()}_model.pkl")
        joblib.dump(model, model_path)
        print(f"{model_name} model saved to {model_path}")

        # 8. Generate Predictions
        print(f"Generating predictions for {model_name} on the test set...")
        predictions = generate_predictions(model, X_test_scaled, label_encoder)
        print("Predictions generated successfully.")

        # 9. Create Submission File
        create_submission_file(test_df, predictions, model_name)

    print("\n--- All Models Processed ---")
    print("Validation Macro F1 Scores:")
    for model_name, score in results.items():
        print(f"- {model_name}: {score}")

    # --- Ensemble Model (VotingClassifier) ---
    print("\n--- Processing Ensemble Model (VotingClassifier) ---")
    
    # Select top 5 models for ensembling based on previous run's performance (or general knowledge)
    # The order here matters for the weights in VOTING_CLASSIFIER_GRID
    top_model_names = ['SVC', 'XGBClassifier', 'KNeighborsClassifier', 'LGBMClassifier', 'RandomForestClassifier']
    
    ensemble_models_dict = {name: trained_models[name] for name in top_model_names if name in trained_models}

    if not ensemble_models_dict:
        print("No top models found for ensembling. Skipping VotingClassifier.")
        results['VotingClassifier'] = 0.0 # Placeholder
    else:
        # Ensure all models in ensemble_models_dict have predict_proba for 'soft' voting
        # For SVC, probability=True must be set in its parameters (already done in config.py)
        
        # Create a base VotingClassifier for GridSearchCV
        base_voting_clf = create_voting_ensemble(ensemble_models_dict)

        print("Performing RandomizedSearchCV for VotingClassifier with 5-fold StratifiedKFold...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)
        random_search_voting = RandomizedSearchCV(estimator=base_voting_clf, 
                                          param_distributions=Config.VOTING_CLASSIFIER_GRID, 
                                          cv=cv, 
                                          scoring='f1_macro', 
                                          n_jobs=-1, 
                                          verbose=1,
                                          n_iter=50,
                                          random_state=Config.RANDOM_STATE)
        
        random_search_voting.fit(X_train, y_train)
        
        print("RandomizedSearchCV complete for VotingClassifier.")
        print(f"Best parameters for VotingClassifier: {random_search_voting.best_params_}")
        print(f"Best macro F1 score for VotingClassifier: {random_search_voting.best_score_:.4f}")

        voting_ensemble = random_search_voting.best_estimator_
        
        print("Evaluating VotingClassifier ensemble on validation set...")
        ensemble_macro_f1 = evaluate_model(voting_ensemble, X_val, y_val)
        print(f"Validation Macro F1 Score for VotingClassifier: {ensemble_macro_f1}")
        results['VotingClassifier'] = ensemble_macro_f1

        # Save Ensemble Model
        ensemble_model_path = os.path.join(Config.MODELS_DIR, "voting_classifier_model.pkl")
        joblib.dump(voting_ensemble, ensemble_model_path)
        print(f"VotingClassifier ensemble model saved to {ensemble_model_path}")

        # Generate Predictions for Ensemble
        print("Generating predictions for VotingClassifier on the test set...")
        ensemble_predictions = generate_predictions(voting_ensemble, X_test_scaled, label_encoder)
        print("Ensemble predictions generated successfully.")

        # Create Submission File for Ensemble
        create_submission_file(test_df, ensemble_predictions, "VotingClassifier")

    # --- Ensemble Model (StackingClassifier) ---
    print("\n--- Processing Ensemble Model (StackingClassifier) ---")

    # Select top 5 models for base estimators (same as for VotingClassifier)
    # Ensure these models are already trained and stored in trained_models
    top_model_names_stacking = ['SVC', 'XGBClassifier', 'KNeighborsClassifier', 'LGBMClassifier', 'RandomForestClassifier']
    
    stacking_base_models_dict = {name: trained_models[name] for name in top_model_names_stacking if name in trained_models}

    if not stacking_base_models_dict:
        print("No top models found for StackingClassifier. Skipping StackingClassifier.")
        results['StackingClassifier'] = 0.0 # Placeholder
    else:
        stacking_ensemble = create_stacking_ensemble(stacking_base_models_dict)
        
        print("Training StackingClassifier ensemble...")
        stacking_ensemble.fit(X_train, y_train)
        print("StackingClassifier ensemble trained successfully.")

        print("Evaluating StackingClassifier ensemble on validation set...")
        stacking_macro_f1 = evaluate_model(stacking_ensemble, X_val, y_val)
        print(f"Validation Macro F1 Score for StackingClassifier: {stacking_macro_f1}")
        results['StackingClassifier'] = stacking_macro_f1

        # Save Stacking Ensemble Model
        stacking_model_path = os.path.join(Config.MODELS_DIR, "stacking_classifier_model.pkl")
        joblib.dump(stacking_ensemble, stacking_model_path)
        print(f"StackingClassifier ensemble model saved to {stacking_model_path}")

        # Generate Predictions for Stacking Ensemble
        print("Generating predictions for StackingClassifier on the test set...")
        stacking_predictions = generate_predictions(stacking_ensemble, X_test_scaled, label_encoder)
        print("Ensemble predictions generated successfully.")

        # Create Submission File for Stacking Ensemble
        create_submission_file(test_df, stacking_predictions, "StackingClassifier")

    print("\n--- All Models Processed ---")
    print("Validation Macro F1 Scores:")
    for model_name, score in results.items():
        print(f"- {model_name}: {score}")

    print("Pipeline finished.")

if __name__ == '__main__':
    main()
