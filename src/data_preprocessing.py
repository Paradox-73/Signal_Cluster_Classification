# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures # Added PolynomialFeatures
from src.config import Config

def load_data(train_path=Config.TRAIN_FILE, test_path=Config.TEST_FILE):
    """
    Loads training and testing data from CSV files.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df, use_polynomial_features=False, polynomial_degree=2):
    """
    Applies preprocessing steps to the data, including feature scaling, optional polynomial feature creation, and target encoding.
    """
    # Separate features and target
    X = train_df[['signal_strength', 'response_level']]
    y = train_df['category']
    X_test = test_df[['signal_strength', 'response_level']]

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Optional: Polynomial Features
    if use_polynomial_features:
        poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
        X_scaled_poly = poly.fit_transform(X_scaled_df)
        X_test_scaled_poly = poly.transform(X_test_scaled_df)

        # Update feature names
        poly_feature_names = poly.get_feature_names_out(X_scaled_df.columns)
        X_scaled_df = pd.DataFrame(X_scaled_poly, columns=poly_feature_names, index=X.index)
        X_test_scaled_df = pd.DataFrame(X_test_scaled_poly, columns=poly_feature_names, index=X_test.index)
        print(f"Applied Polynomial Features with degree {polynomial_degree}. New feature count: {X_scaled_df.shape[1]}")


    # Target Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_scaled_df, y_encoded, X_test_scaled_df, label_encoder, scaler

def split_data(X, y):
    """
    Splits the training data into training and validation sets.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    return X_train, X_val, y_train, y_val
